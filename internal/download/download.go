// Package download fetches the three MaleCNS v1.0 flat-connectome tables (~1.1 GB).
//
// Files are split into byte ranges fetched by parallel goroutines, each writing its part
// at the right offset of the output file. Progress is recorded in a small sidecar file,
// so an interrupted download resumes. The HTTP client honours HTTPS_PROXY; Google Cloud
// Storage is geo-blocked in some regions.
package download

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	bucket = "flyem-male-cns"
	prefix = "v1.0/connectome-data/flat-connectome/"

	Annotations       = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
	Neurotransmitters = "body-neurotransmitters-male-cns-v1.0.feather"
	Weights           = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

	parts = 16
)

// Files lists every table the project needs.
var Files = []string{Annotations, Neurotransmitters, Weights}

var client = &http.Client{Transport: &http.Transport{Proxy: http.ProxyFromEnvironment}}

// All downloads every missing or incomplete file into dir.
func All(dir string) error {
	for _, name := range Files {
		if err := Fetch(dir, name); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
	}
	return nil
}

func remoteSize(name string) (int64, error) {
	u := fmt.Sprintf("https://storage.googleapis.com/storage/v1/b/%s/o/%s?fields=size",
		bucket, url.PathEscape(prefix+name))
	resp, err := client.Get(u)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("size lookup: %s", resp.Status)
	}
	var body struct {
		Size string `json:"size"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return 0, err
	}
	return strconv.ParseInt(body.Size, 10, 64)
}

// Fetch downloads one file with parallel ranged requests, verified against the bucket's size.
func Fetch(dir, name string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	total, err := remoteSize(name)
	if err != nil {
		return err
	}
	dest := filepath.Join(dir, name)
	progressPath := dest + ".parts"
	done := make([]bool, parts)
	if st, err := os.Stat(dest); err == nil && st.Size() == total {
		if _, err := os.Stat(progressPath); os.IsNotExist(err) {
			fmt.Printf("ok  %s\n", name)
			return nil
		}
	}
	if js, err := os.ReadFile(progressPath); err == nil {
		_ = json.Unmarshal(js, &done)
	}

	f, err := os.OpenFile(dest, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := f.Truncate(total); err != nil {
		return err
	}

	var (
		wg       sync.WaitGroup
		mu       sync.Mutex
		received atomic.Int64
		firstErr error
	)
	chunk := (total + parts - 1) / parts
	stop := make(chan struct{})
	go reportProgress(name, total, &received, stop)

	for p := range parts {
		lo, hi := int64(p)*chunk, min(int64(p+1)*chunk, total)-1
		if done[p] || lo > hi {
			received.Add(max(hi-lo+1, 0))
			continue
		}
		wg.Go(func() {
			err := fetchRange(f, name, lo, hi, &received)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				if firstErr == nil {
					firstErr = err
				}
				return
			}
			done[p] = true
			js, _ := json.Marshal(done)
			_ = os.WriteFile(progressPath, js, 0o644)
		})
	}
	wg.Wait()
	close(stop)
	if firstErr != nil {
		return fmt.Errorf("%w (run again to resume)", firstErr)
	}
	fmt.Printf("\rok  %s%s\n", name, strings.Repeat(" ", 30))
	return os.Remove(progressPath)
}

// fetchRange downloads bytes lo..hi, retrying a few times from where it stopped.
func fetchRange(f *os.File, name string, lo, hi int64, received *atomic.Int64) error {
	var lastErr error
	for attempt := range 5 {
		if attempt > 0 {
			time.Sleep(time.Duration(attempt) * 2 * time.Second)
		}
		req, err := http.NewRequest(http.MethodGet,
			fmt.Sprintf("https://storage.googleapis.com/%s/%s%s", bucket, prefix, name), nil)
		if err != nil {
			return err
		}
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", lo, hi))
		resp, err := client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		if resp.StatusCode != http.StatusPartialContent {
			resp.Body.Close()
			lastErr = fmt.Errorf("range request: %s", resp.Status)
			continue
		}
		n, err := io.Copy(io.NewOffsetWriter(f, lo), &countingReader{resp.Body, received})
		resp.Body.Close()
		lo += n
		if err == nil && lo > hi {
			return nil
		}
		lastErr = err
	}
	return lastErr
}

type countingReader struct {
	r io.Reader
	n *atomic.Int64
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n.Add(int64(n))
	return n, err
}

func reportProgress(name string, total int64, received *atomic.Int64, stop <-chan struct{}) {
	tick := time.NewTicker(time.Second)
	defer tick.Stop()
	for {
		select {
		case <-stop:
			return
		case <-tick.C:
			fmt.Printf("\r    %s  %5.1f%% of %.0f MB", name[:min(len(name), 40)],
				100*float64(received.Load())/float64(total), float64(total)/1e6)
		}
	}
}
