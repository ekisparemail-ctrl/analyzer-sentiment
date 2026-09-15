# To-Do: Analytics Backend

## Alur Kerja

1. **Menerima input** — plain teks + URL video dari Scrapper Backend (project terpisah, dikerjakan oleh developer lain).
2. **Download video** berdasarkan URL yang diberikan.
3. **Meringkas video** (long video summarization) menggunakan tool open source — tidak perlu membangun dari awal.
4. **Hasil antara**: plain teks + ringkasan video, siap dianalisis.
5. **Analisis sentimen** menggunakan kombinasi VLM + LLM.
6. **Kirim hasil** kembali ke Scrapper Backend.

## Catatan

Scrapper Backend mengirimkan data satu per satu (bukan batch). Kita proses satu per satu, dan mengembalikan hasil satu per satu juga — mengikuti pola pengiriman yang sama.
