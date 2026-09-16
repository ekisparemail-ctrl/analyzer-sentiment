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

### Model lokal (transformers, faster-whisper) — pisahkan "load" dari "call"

`ai/vlm_local.py` mengikuti pola yang sama, tapi satu tingkat lebih dalam:
`load_local_vlm(model_id)` (panggilan `from_pretrained()` yang asli — lambat, butuh
bobot model asli) **tidak** ditest, tapi `describe_images(vlm, ...)` (logic
glue: membangun format chat multi-image, memanggil `generate()`, decode hasil) **ditest
penuh** dengan `model`/`processor` palsu (`MagicMock`), bukan lewat dependency-injection
callable seperti `AnalyzerModels` di atas — karena bentuk API `transformers` (dict
input, tensor output, slicing berdasarkan panjang prompt) sendiri yang mau divalidasi:

```python
processor = MagicMock()
processor.apply_chat_template.return_value = "PROMPT_TEXT"
processor.return_value = {"input_ids": torch.zeros((1, 3), dtype=torch.long)}
processor.batch_decode.return_value = ["hasil deskripsi"]

model = MagicMock()
model.generate.return_value = torch.zeros((1, 8), dtype=torch.long)
```

Pakai tensor `torch` asli (bukan `MagicMock` polos) untuk bagian yang di-slice
(`output_ids[:, inputs["input_ids"].shape[1]:]`) — ini membuktikan logic
pemotongan prompt-token benar-benar jalan, bukan cuma "tidak error karena semuanya
mock". Gambar uji dibuat lewat `PIL.Image` asli (`Image.new(...).save(buf, "JPEG")`),
bukan bytes acak, karena `describe_images()` benar-benar memanggil
`Image.open()`/`.convert("RGB")`.

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

## Pengujian End-to-End (manual)

Test otomatis di `tests/` sengaja **tidak** menyentuh Kafka, LLM, atau model
VLM/whisper yang sungguhan (lihat spec §2 "Out of scope": *"End-to-end integration
tests against a real Kafka cluster or real model weights (VLM/whisper)"*). Semua
boundary itu di-mock. Karena itu, verifikasi bahwa seluruh alur benar-benar nyambung
ujung-ke-ujung harus dilakukan manual terhadap infrastruktur asli, mengikuti langkah
di bawah.

### Alur yang diuji

```
Scrapper BE            Analytics Backend (project ini, semua lokal)          Mac Studio
    │                              │                                            │
    │ 1. publish request           │                                            │
    │   (Kafka: topic request)     │                                            │
    ├─────────────────────────────►│                                            │
    │                              │ 2. consumer.poll_request()                 │
    │                              │    -> AnalysisRequest                      │
    │                              │                                            │
    │                              │ 3. jika video_url ada:                     │
    │                              │    a. download_video() (yt-dlp)            │
    │                              │    b. extract_frames() (ffmpeg, lokal)     │
    │                              │    c. describe_images() (ai/vlm_local.py,  │
    │                              │       in-process, CPU, tanpa jaringan)      │
    │                              │    d. transcribe() (faster-whisper,        │
    │                              │       lokal, CPU)                          │
    │                              │                                            │
    │                              │ 4. summarize_video() ──────────────────────┤──► LLM (HTTP,
    │                              │    (ai/llm_client.py)                      │     LM Studio)
    │                              │◄────────────────────────────────────────────┤ video_summary
    │                              │                                            │
    │                              │ 5. gabungkan text + video_summary,         │
    │                              │    analyze_sentiment() ────────────────────┤──► LLM (HTTP)
    │                              │◄────────────────────────────────────────────┤ sentiment/
    │                              │                                            │  emotion/motivation
    │                              │ 6. producer.publish(AnalysisResult)        │
    │  7. consume hasil            │    lalu consumer.commit(msg)               │
    │   (Kafka: topic response)    │                                            │
    │◄─────────────────────────────┤                                            │
```

Setiap tahap (2-6) sudah ditest terisolasi lewat unit test (lihat bagian di atas).
Pengujian manual ini memverifikasi bahwa tahap-tahap itu benar saat dirangkai dengan
infrastruktur sungguhan, plus perilaku yang cuma muncul di integrasi nyata: konten
`.env` yang valid, `ffmpeg` yang benar-benar terpasang, model VLM/whisper yang
benar-benar ter-download dan bisa jalan di CPU mesin ini, koneksi jaringan ke Mac
Studio untuk LLM saja, dan format pesan Kafka yang sungguhan cocok dengan
`AnalysisRequest`/`AnalysisResult`.

### Prasyarat & konfigurasi tambahan

Sebelum menjalankan pengujian manual ini, pastikan semua ini tersedia:

- **Python 3.11+** dengan virtualenv sudah dibuat dan `requirements.txt` +
  `requirements-dev.txt` ter-install (`pip install -r requirements-dev.txt`).
- **`ffmpeg` dan `ffprobe` terpasang dan ada di PATH** (`ffmpeg -version` harus
  berhasil dari shell yang sama tempat service dijalankan). Ini dipakai
  `video/frames.py` (ekstraksi frame) dan `faster-whisper` (ekstraksi audio) — bukan
  dependency Python, jadi tidak ter-install lewat `pip`.
- **Kafka broker yang bisa diakses** dari mesin ini, dengan topic request dan
  response sudah dibuat (atau broker mengizinkan auto-create topic). Nama topic
  default: `scrapper-to-analysis` (request, `KAFKA_SCRAPPER_TOPIC`) dan
  `analysis-to-scrapper` (response, `KAFKA_RESULT_TOPIC`) — bisa diubah lewat env,
  lihat `.env.example`.
- **LM Studio berjalan di Mac Studio** dengan model LLM sudah di-load, dan endpoint-nya
  bisa diakses dari mesin ini (`LLM_BASE_URL`, format OpenAI-compatible, contoh:
  `http://<ip-mac-studio>:1234/v1`). Ini satu-satunya bagian pipeline yang masih
  memanggil Mac Studio — VLM sekarang berjalan lokal (lihat poin berikutnya).
- **VLM berjalan in-process, CPU-only, di mesin ini sendiri** — tidak ada endpoint
  atau konfigurasi jaringan yang perlu disiapkan devops. `VLM_MODEL_ID` (default
  `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`) akan otomatis ter-download dari
  HuggingFace Hub ke cache lokal saat pertama kali `load_models()` dipanggil (butuh
  koneksi internet sekali di awal, dan dependency `torch`+`transformers`+`pillow`
  ter-install — lihat `requirements.txt`). Karena CPU-only, inferensi VLM bisa
  memakan waktu signifikan per frame batch — belum ada angka latency resmi untuk
  model default ini; kalau lambat, pertimbangkan mengecilkan `MAX_FRAMES` dulu
  sebelum mengganti model (spec §9).
- **Model `faster-whisper`** juga otomatis ter-download ke cache lokal saat pertama
  kali `load_models()` dipanggil (ukuran tergantung `WHISPER_MODEL_SIZE`, default
  `base`). Untuk pengujian pertama kali, jalankan service dan tunggu sampai **kedua**
  model (VLM + whisper) selesai di-download/di-load sebelum mengirim pesan test —
  cek log startup.
- **File `.env`** disalin dari `.env.example` (di root project, sejajar dengan
  `src/`) dan diisi nilai asli (bukan placeholder): `KAFKA_BOOTSTRAP_SERVERS`,
  `LLM_BASE_URL`, `LLM_MODEL`, `VLM_MODEL_ID` (opsional, ada default), dan lainnya
  sesuai kebutuhan. `Settings` (`src/config.py`) otomatis membaca file `.env` ini kalau
  ada di working directory saat `src/main.py` dijalankan (`env_file=".env"` di
  `SettingsConfigDict`) — tidak perlu export manual ke environment. Kalau
  sebuah env var **juga** di-set langsung di environment proses, nilai
  environment itu yang menang (bukan isi `.env`). *(Riwayat: sebelum
  perbaikan di `docs/issue-findings.md`, `Settings` cuma baca `os.environ`
  dan mengabaikan `.env` sama sekali — menjalankan `src/main.py` langsung
  setelah menyalin `.env.example` akan gagal dengan `ValidationError: Field
  required` untuk `kafka_bootstrap_servers`/`llm_base_url`/`llm_model` walau
  filenya sudah terisi benar. Ini sudah diperbaiki; disebut di sini supaya
  gejalanya dikenali kalau muncul lagi di versi lama.)*
- **URL video contoh** yang valid dan bisa diakses `yt-dlp` (TikTok/Instagram/
  Facebook/Twitter-X), durasi di bawah ~10 menit sesuai asumsi spec §2/§9.

### Langkah pengujian manual

1. **Jalankan service**:
   ```bash
   .venv/Scripts/python src/main.py    # Windows
   .venv/bin/python src/main.py        # macOS/Linux
   ```
   Tunggu sampai log menunjukkan model whisper selesai dimuat dan consumer siap
   polling (tidak ada log error saat startup). Baris log pertama
   (`Starting Analytics Backend (kafka_bootstrap_servers=...)`) mencetak
   alamat broker dan nama topic yang benar-benar dipakai — **cek baris ini
   dulu** kalau service tampak diam/retry terus tanpa progres, untuk
   memastikan `KAFKA_BOOTSTRAP_SERVERS` di `.env` memang menunjuk ke broker
   asli (`172.16.16.100:21000`), bukan `localhost:9092` atau nilai lain yang
   tidak sengaja ketinggalan dari testing lokal. Kalau memang salah alamat,
   `poll()` akan retry tanpa henti (perilaku normal, bukan bug) dan hanya
   terlihat lewat log mentah `rdkafka#...FAIL` yang cukup teknis.

   **Gejala berbeda yang perlu dibedakan:** kalau baris `Starting Analytics
   Backend (kafka_bootstrap_servers=...)` sudah menunjukkan alamat broker
   yang **benar**, tapi baris `FAIL` di bawahnya tetap menyebut alamat
   **lain** (mis. `localhost:9092`/`127.0.0.1:9092`) — terutama kalau itu
   muncul untuk `GroupCoordinator` DAN untuk producer sekaligus — itu bukan
   masalah `.env`/konfigurasi kita sama sekali. Itu artinya broker Kafka
   itu sendiri meng-advertise alamat yang salah ke semua client (klasik
   `advertised.listeners` broker ter-set ke `localhost`, bukan alamat yang
   benar-benar reachable) — laporkan ke devops/pengelola Kafka-nya, bukan
   diutak-atik di sisi project ini.
   Menghentikan service dengan Ctrl-C akan shutdown bersih (log "Received
   interrupt signal, shutting down gracefully." lalu keluar), bukan
   menampilkan traceback mentah.

2. **Kirim satu pesan test** ke topic request. Payload harus cocok dengan bentuk asli
   pesan Scrapper Backend, `NormalizedData` (`src/messaging/scrapper_dto.py`,
   camelCase) — **bukan** `AnalysisRequest` langsung, itu bentuk internal setelah
   diterjemahkan `request_from_normalized_data()`:
   ```json
   {
     "id": "manual-test-001",
     "platform": "twitter",
     "type": "POST",
     "message": "Contoh teks postingan untuk pengujian manual",
     "videoUrl": "https://contoh.com/path/ke/video.mp4",
     "imageUrl": null,
     "authorUsername": "contoh_user",
     "authorName": "Contoh User",
     "views": null,
     "likes": 0,
     "repliesCount": 0,
     "uploadedAt": 1700000000,
     "commentTo": null
   }
   ```
   `message` dan `videoUrl` boleh salah satu `null`/dihilangkan untuk menguji jalur
   teks-saja atau video-saja secara terpisah — lihat kombinasi yang relevan di
   `tests/pipeline/test_analyze.py` sebagai referensi skenario yang perlu dicoba
   (teks+video, video-saja, teks-saja, video gagal, dst), dan
   `tests/messaging/test_scrapper_dto.py`'s
   `test_parses_real_payload_captured_from_scrapper_be` untuk contoh payload asli
   yang pernah tertangkap dari produksi.

   Cara paling sederhana mengirim pesan tanpa tooling tambahan: skrip Python kecil
   pakai `confluent-kafka` yang sudah ada di `venv` project ini:
   ```python
   import json
   from confluent_kafka import Producer

   p = Producer({"bootstrap.servers": "<isi sesuai KAFKA_BOOTSTRAP_SERVERS>"})
   payload = {
       "id": "manual-test-001",
       "platform": "twitter",
       "type": "POST",
       "message": "Contoh teks postingan untuk pengujian manual",
       "videoUrl": "https://contoh.com/path/ke/video.mp4",
   }
   p.produce(
       "scrapper-to-analysis", key=payload["id"].encode(), value=json.dumps(payload).encode()
   )
   p.flush(10)
   ```
   Kalau broker sudah punya `kafka-console-producer.sh` terpasang, itu juga bisa
   dipakai langsung (paste payload JSON sebagai satu baris).

3. **Amati log service** — setiap tahap alur di atas seharusnya meninggalkan jejak log
   (skip pesan malformed, error Kafka-level, kegagalan video yang di-degrade, status
   akhir tiap item). Untuk video: proses download → ekstraksi frame → inferensi VLM
   lokal → transkripsi whisper akan makan waktu beberapa detik hingga puluhan detik
   tergantung ukuran video dan kecepatan CPU mesin ini (VLM dan whisper keduanya
   jalan lokal sekarang, bukan cuma whisper).

4. **Konsumsi topic response** untuk melihat hasilnya:
   ```python
   from confluent_kafka import Consumer

   c = Consumer({
       "bootstrap.servers": "<isi sesuai KAFKA_BOOTSTRAP_SERVERS>",
       "group.id": "manual-test-consumer",
       "auto.offset.reset": "earliest",
   })
   c.subscribe(["analysis-to-scrapper"])
   msg = c.poll(30)
   print(msg.value().decode())
   ```
   Verifikasi bentuk `AnalysisResult` (`src/schemas.py`): `id` cocok dengan yang
   dikirim, `status` sesuai ekspektasi (`ok`/`partial`/`failed`), `video_summary`
   terisi kalau video diproses, dan `sentiment`/`emotion`/`motivation` terisi kalau
   `status` bukan `failed`.

5. **Verifikasi offset commit**: kirim pesan lagi dengan `id` yang sama, restart
   service, dan pastikan pesan yang **sudah** berhasil diproses+publish sebelumnya
   **tidak** diproses ulang (offset ter-commit) — sementara pesan yang servicenya
   mati di tengah proses (matikan paksa sebelum publish selesai) **akan** diproses
   ulang saat service dinyalakan lagi (bukti garansi *at-least-once*: offset baru
   commit setelah publish sukses, lihat `src/main.py`'s `run_once()`).

### Kapan wajib menjalankan ini

- Sebelum deploy pertama kali ke lingkungan yang menjalankan Kafka/LM Studio/VLM
  sungguhan.
- Setelah mengganti `VLM_MODEL_ID` ke model lain (mis. hasil spike latency
  menunjukkan model default terlalu lambat/kurang akurat) — ulangi langkah di atas
  untuk memastikan jalur video tetap `status: "ok"`.
- Setelah perubahan apa pun ke `src/main.py`, `src/messaging/`, atau kontrak pesan di
  `src/schemas.py` — perubahan di area ini paling berisiko lolos dari unit test
  (yang semuanya mocked) tapi patah di integrasi nyata.
