# Testing Guidelines

Panduan ini merangkum konvensi testing yang dipakai di project `analyzer_sentiment`,
berdasarkan pola yang sudah terbukti jalan (dan beberapa kesalahan yang sempat terjadi
lalu diperbaiki) selama implementasi Analytics Backend. Ikuti ini untuk kode baru supaya
gaya test tetap konsisten di seluruh codebase.

## Prinsip dasar

- **TDD wajib untuk logic baru**: tulis test yang gagal dulu (RED), baru implementasi
  minimal supaya lolos (GREEN). Jangan menulis implementasi lalu menyusulkan test.
- **Test menguji perilaku, bukan detail implementasi.** Assert pada input/output dan
  efek yang teramati (exception yang dilempar, fungsi yang dipanggil dengan argumen
  tertentu), bukan pada struktur internal yang kebetulan.
- **Bug fix wajib disertai regression test** yang gagal sebelum fix dan lolos sesudahnya.
- **Sebelum menandai pekerjaan selesai**, tiga perintah ini harus bersih:
  ```bash
  pytest -v
  ruff check .
  mypy src
  ```
  Ketiganya adalah gate resmi project ini — jangan klaim "selesai" tanpa menjalankan
  ketiganya dan melihat outputnya secara langsung (jangan asumsi dari ingatan run
  sebelumnya).

## Struktur file test

Test mengikuti struktur `src/` satu-satu:

```
src/config.py               -> tests/test_config.py
src/schemas.py               -> tests/test_schemas.py
src/main.py                  -> tests/test_main.py
src/video/downloader.py      -> tests/video/test_downloader.py
src/video/analyzer.py        -> tests/video/test_analyzer.py
src/video/frames.py          -> tests/video/test_frames.py
src/ai/llm_client.py         -> tests/ai/test_llm_client.py
src/ai/vlm_client.py         -> tests/ai/test_vlm_client.py
src/pipeline/analyze.py      -> tests/pipeline/test_analyze.py
src/messaging/producer.py    -> tests/messaging/test_producer.py
src/messaging/consumer.py    -> tests/messaging/test_consumer.py
```

**Jangan tambahkan `tests/__init__.py` atau `tests/<subdir>/__init__.py`.** Ini bukan
gaya pribadi — ini sudah pernah dicoba dua kali di project ini dan keduanya harus
dibatalkan setelah review, karena:

1. Tidak diperlukan sama sekali — `pyproject.toml` sudah set `pythonpath = ["src"]`,
   dan pytest mode "prepend" (default) sudah bisa discover semua test file tanpa
   `__init__.py` di manapun.
2. Menambahkannya justru mengubah *import identity* file test yang sudah ada (jadi
   `tests.test_config` alih-alih `test_config`), efek samping ke file yang tidak
   sedang disentuh.
3. Kalau ditambahkan cuma di satu subdirektori, itu bikin struktur `tests/` jadi tidak
   konsisten (sebagian pakai package, sebagian tidak) tanpa alasan fungsional.

Kalau `pytest -v` gagal mendiscover test baru kamu, penyebabnya hampir pasti bukan
"perlu `__init__.py`" — cek dulu nama file (`test_*.py`), lokasi (di bawah `tests/`),
dan `pythonpath` di `pyproject.toml`.

## Pola mocking per jenis dependency

### HTTP client (httpx) — pakai `respx`

Semua client HTTP (`ai/llm_client.py`, `ai/vlm_client.py`) di-test dengan `respx`,
bukan `unittest.mock` manual. Ini memberi verifikasi request-level yang sungguhan
(body, header, jumlah call) tanpa koneksi network nyata.

```python
import httpx
import respx

@respx.mock
def test_describe_images_sends_prompt_and_images_and_returns_text() -> None:
    route = respx.post("http://localhost:8001/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "..."}}]})
    )
    result = describe_images(_config(), "prompt", ["data:image/png;base64,AAA"])
    assert result == "..."
    assert route.called
```

**Untuk retry logic**, selalu assert `route.call_count` — jangan cuma assert hasil
akhirnya benar. Pola `respx`'s `side_effect` sebagai list response berguna untuk
simulasi "gagal N kali lalu berhasil":

```python
@respx.mock
def test_retries_then_succeeds() -> None:
    route = respx.post(url).mock(
        side_effect=[httpx.Response(500), httpx.Response(200, json={...})]
    )
    result = summarize_video(_config(), "vlm summary", None)
    assert route.call_count == 2  # <- wajib, ini yang membuktikan retry benar-benar jalan
```

Pernah terjadi test retry-exhaustion yang tidak assert `call_count` sama sekali —
lolos meski retry-nya diam-diam tidak jalan (cuma 1x percobaan). Jangan ulangi ini.

Set `retry_backoff_seconds=0.0` di config test supaya `time.sleep()` antar-retry tidak
memperlambat suite.

### subprocess (yt-dlp, ffmpeg) — patch dengan fake yang punya efek nyata

Untuk `video/downloader.py` (yt-dlp) dan `video/frames.py` (ffmpeg/ffprobe), jangan
cuma assert bahwa fungsi mock dipanggil — buat fake yang benar-benar menghasilkan
efek (menulis file) supaya kode setelahnya (baca file, encode base64, dst) betul-betul
teruji, bukan cuma "mock dipanggil dengan argumen X":

```python
def _fake_run_factory(frame_count: int):
    def _fake_run(cmd, check, capture_output, text=False):
        if cmd[0] == "ffprobe":
            return MagicMock(stdout=json.dumps({"format": {"duration": "10.0"}}))
        if cmd[0] == "ffmpeg":
            frames_dir = os.path.dirname(cmd[-1])
            for i in range(frame_count):
                with open(os.path.join(frames_dir, f"frame-{i:03d}.jpg"), "wb") as f:
                    f.write(b"\xff\xd8\xff\xe0fakejpeg")
            return MagicMock(returncode=0)
        raise AssertionError(f"unexpected command: {cmd}")
    return _fake_run

def test_extract_frames_returns_base64_data_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("video.frames.subprocess.run", _fake_run_factory(3))
    result = extract_frames("/tmp/video.mp4", max_frames=5)
    assert len(result) == 3
```

Untuk logic komputasi (misal `fps = max_frames / duration`), assert langsung terhadap
argv yang benar-benar dikirim ke subprocess — jangan cuma test hasil akhirnya, karena
itu bisa lolos meski rumusnya salah asal jumlah frame yang dihasilkan tetap sama.

### Kafka (confluent-kafka) — patch nama yang di-import ke modul, bukan library aslinya

```python
monkeypatch.setattr("messaging.producer.Producer", fake_producer_class)
monkeypatch.setattr("messaging.consumer.Consumer", fake_consumer_class)
```

Bukan `confluent_kafka.Producer`/`confluent_kafka.Consumer` — patch di titik pemakaian
(`from confluent_kafka import Producer` di `messaging/producer.py` berarti nama
`Producer` sekarang hidup di namespace `messaging.producer`).

**Delivery callback confluent-kafka bersifat async secara default** — kalau kode
sedang test punya perilaku yang bergantung pada callback (delivery report, commit
result), buat mock `produce()`/`commit()` memanggil callback-nya secara sinkron lewat
`side_effect`, supaya test benar-benar melewati jalur callback tersebut:

```python
def fake_produce(topic, key, value, callback):
    callback(None, MagicMock())  # simulasikan delivery sukses secara sinkron

fake_producer_instance.produce.side_effect = fake_produce
```

**Set eksplisit return value mock kalau kode membaca return value-nya.** Ini pelajaran
mahal dari project ini: `Producer.flush(timeout)` mengembalikan jumlah pesan yang
masih pending. Kode awalnya membuang return value ini (bug fail-open — lihat
"Kelas bug yang harus selalu dicek" di bawah). Setelah diperbaiki, test yang sudah ada
harus di-update supaya set `flush.return_value = 0` secara eksplisit — kalau tidak,
`MagicMock()` default (truthy, bukan `0`) akan memicu error palsu di test yang
seharusnya menguji jalur sukses.

### Dependency injection (bukan mocking library) — untuk orchestrator

`pipeline/analyze.py` dan `video/analyzer.py`'s `analyze_video()` sengaja didesain
menerima dependency lewat parameter (`AnalyzeDependencies`, `AnalyzerModels`) berisi
callable polos, bukan lewat import langsung ke library pihak ketiga. Ini bikin test-nya
tidak perlu mocking sama sekali — cukup lempar lambda/closure biasa:

```python
def _deps(**overrides: object) -> AnalyzeDependencies:
    defaults: dict = dict(
        download_video=lambda url, dest_dir: "/tmp/video.mp4",
        analyze_video=lambda path, prompt, max_frames, max_tokens, include_transcript:
            VideoAnalysis(summary="s", transcript="t", transcript_segments=[]),
        summarize_video=lambda vlm_summary, transcript: "video summary",
        analyze_sentiment=lambda content: _sentiment_analysis(),
    )
    defaults.update(overrides)
    return AnalyzeDependencies(**defaults)
```

Pola ini yang membuat `video/analyzer.py`'s `load_models()` bisa dipisah bersih dari
`analyze_video()`: `load_models()` yang benar-benar memuat model whisper lokal (lambat,
butuh model asli) sengaja **tidak** ditest otomatis, sementara `analyze_video()` (murni
logic orkestrasi) di-test penuh lewat `AnalyzerModels` palsu. Kalau menambah dependency
baru yang "berat" (model, koneksi eksternal), pertimbangkan pola yang sama: pisahkan
"cara memuat/memanggil resource berat" dari "logic yang memakainya".

## Kelas bug yang harus selalu dicek: return value / state ambigu yang didiskon diam-diam

Selama project ini, tiga bug nyata (satu ditemukan security scan otomatis, dua oleh
review manual) punya bentuk yang sama: **memanggil API pihak ketiga yang bisa gagal
secara ambigu, lalu mengabaikan sinyal kegagalannya** — sehingga kondisi
tidak-jelas/timeout diam-diam diperlakukan sebagai sukses ("fail-open").

Contoh yang sudah ditemukan di project ini:
- `Producer.flush(timeout)` mengembalikan jumlah pesan yang masih pending — return
  value ini sempat dibuang begitu saja.
- `Consumer.commit(msg)` secara default `asynchronous=True` — sukses/gagalnya cuma
  dilaporkan lewat callback `on_commit` yang tidak pernah dikonfigurasi, jadi hasilnya
  hilang total.
- `msg.value()` bisa mengembalikan `None` (misal tombstone message) — memanggil
  `.decode()` langsung di atasnya akan crash dengan `AttributeError`, exception yang
  tidak tertangkap oleh handler "pesan malformed".

**Saat menulis atau mereview kode yang memanggil API eksternal (Kafka, HTTP, subprocess,
library apa pun)**, selalu tanyakan: *"apa return value atau exception yang mungkin
saya lewatkan begitu saja?"* — dan tulis test yang secara eksplisit memaksa jalur
ambigu/gagal itu (bukan cuma jalur sukses telanjang), untuk membuktikan kode
menanganinya, bukan mengabaikannya.

## Cakupan test yang wajib untuk kode baru

Untuk setiap fungsi/module baru yang punya percabangan (bukan cuma data class), pastikan
ada test untuk:

1. Jalur sukses (happy path).
2. Setidaknya satu jalur gagal per exception yang ditangkap secara eksplisit — bukan
   cuma exception generik.
3. Kalau ada retry: jalur "gagal lalu berhasil" DAN jalur "gagal terus sampai retry
   habis" — keduanya, dengan assert `call_count`.
4. Kalau ada graceful degrade (mis. hasil parsial ketika satu bagian gagal): pastikan
   informasi dari kegagalan itu tidak hilang di jalur akhir (lihat contoh
   `tests/pipeline/test_analyze.py`'s test untuk kombinasi "video gagal DAN sentiment
   analysis juga gagal" — keduanya harus tetap muncul di pesan error akhir).
5. Modul yang punya "kembaran" (mis. `ai/llm_client.py` dan `ai/vlm_client.py` sama-sama
   HTTP client dengan retry) harus dites setara — jangan sampai satu punya assertion
   yang lebih ketat dari yang lain untuk logic yang identik. Ini pernah kejadian:
   `vlm_client.py`'s test retry tidak seketat `llm_client.py`'s, dan baru ketahuan di
   final review, bukan di review per-modul.

## Type hints di kode test

Fungsi test dan helper di dalam file test tetap harus punya type hint pada parameter
dan return type, mengikuti standar "semua fungsi punya type hint" — meskipun `mypy src`
(gate resmi project ini) tidak memeriksa folder `tests/`, jadi pelanggaran di sini tidak
akan gagal di CI. Konsistensi tetap penting untuk keterbacaan.

```python
def test_analyze_text_and_video_produces_ok_status() -> None:
    ...

def _fake_run(cmd: list[str], check: bool, capture_output: bool, text: bool = False) -> MagicMock:
    ...
```

## Menjalankan test

```bash
# seluruh suite
pytest -v

# satu file/fungsi spesifik saat iterasi
pytest tests/pipeline/test_analyze.py -v
pytest tests/pipeline/test_analyze.py::test_analyze_text_and_video_produces_ok_status -v

# gate lengkap sebelum menganggap pekerjaan selesai
pytest -v && ruff check . && mypy src
```

`docs/references/server.py` (file referensi dari supervisor, bukan kode yang kita
tulis) sengaja dikecualikan dari `ruff check .` lewat `extend-exclude = ["docs/"]` di
`pyproject.toml` — jangan format ulang file itu, dan jangan hapus exclude-nya.
