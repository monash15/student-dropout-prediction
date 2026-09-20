"""
label_pita.py
=============
Penerjemah kode dataset menjadi label berbahasa Indonesia untuk kartu skor.

Kartu skor hanya berguna kalau bisa dibaca tanpa penerjemah. Baris "Course = 9500,
+61 poin" tidak berarti apa apa bagi dosen wali, sedangkan "Program studi
Keperawatan, +61 poin" langsung dapat dipakai. Modul ini menyimpan seluruh
padanannya dalam satu kamus bersarang, lalu menyediakan dua fungsi pembaca:
``terjemahkan`` untuk isi sel dan ``nama_variabel`` untuk judul kolom.

Acuan kode mengikuti dokumentasi dataset Predict Students Dropout and Academic
Success pada UCI Machine Learning Repository, sumber yang sama dengan berkas yang
dibagikan Dicoding.
"""

# ---------------------------------------------------------------------------
# Padanan kode untuk setiap kolom kategori
# ---------------------------------------------------------------------------
TERJEMAHAN = {
    "Course": {
        33: "Teknologi Bahan Bakar Nabati",
        171: "Animasi dan Desain Multimedia",
        8014: "Kerja Sosial kelas malam",
        9003: "Agronomi",
        9070: "Desain Komunikasi Visual",
        9085: "Perawat Hewan",
        9119: "Teknik Informatika",
        9130: "Peternakan Kuda",
        9147: "Manajemen",
        9238: "Kerja Sosial",
        9254: "Pariwisata",
        9500: "Keperawatan",
        9556: "Higiene Gigi",
        9670: "Manajemen Iklan dan Pemasaran",
        9773: "Jurnalistik dan Komunikasi",
        9853: "Pendidikan Dasar",
        9991: "Manajemen kelas malam",
    },
    "Application_mode": {
        1: "Seleksi umum tahap 1",
        2: "Aturan khusus 612/93",
        5: "Kuota wilayah Azores",
        7: "Lulusan perguruan tinggi lain",
        10: "Aturan khusus 854-B/99",
        15: "Pendaftar internasional",
        16: "Kuota wilayah Madeira",
        17: "Seleksi umum tahap 2",
        18: "Seleksi umum tahap 3",
        26: "Aturan 533-A/99 kurikulum berbeda",
        27: "Aturan 533-A/99 institusi lain",
        39: "Jalur usia di atas 23 tahun",
        42: "Alih kampus",
        43: "Alih program studi",
        44: "Pemegang diploma vokasi teknologi",
        51: "Alih kampus sekaligus program studi",
        53: "Pemegang diploma siklus pendek",
        57: "Alih kampus jalur internasional",
    },
    "Marital_status": {
        1: "Lajang",
        2: "Menikah",
        3: "Ditinggal wafat pasangan",
        4: "Bercerai",
        5: "Tinggal bersama tanpa nikah",
        6: "Berpisah secara hukum",
    },
    "Previous_qualification": {
        1: "Tamat sekolah menengah",
        2: "Sarjana",
        3: "Diploma perguruan tinggi",
        4: "Magister",
        5: "Doktor",
        6: "Sempat kuliah tanpa lulus",
        9: "Kelas 12 tidak tamat",
        10: "Kelas 11 tidak tamat",
        12: "Setara kelas 11 jalur lain",
        14: "Tamat kelas 10",
        15: "Kelas 10 tidak tamat",
        19: "Pendidikan dasar tahap 3",
        38: "Pendidikan dasar tahap 2",
        39: "Kursus keahlian teknologi",
        40: "Sarjana siklus pertama",
        42: "Kursus teknik tingkat tinggi",
        43: "Magister siklus kedua",
    },
    "Daytime_evening_attendance": {0: "Kelas malam", 1: "Kelas siang"},
    "Gender": {0: "Perempuan", 1: "Laki laki"},
    "Displaced": {0: "Tinggal di kota asal", 1: "Merantau"},
    "Debtor": {0: "Tidak menunggak", 1: "Menunggak"},
    "Tuition_fees_up_to_date": {0: "Uang kuliah tertunggak", 1: "Uang kuliah lancar"},
    "Scholarship_holder": {0: "Bukan penerima beasiswa", 1: "Penerima beasiswa"},
    "International": {0: "Warga negara sendiri", 1: "Warga negara asing"},
    "Educational_special_needs": {0: "Tanpa kebutuhan khusus", 1: "Berkebutuhan khusus"},
}

# ---------------------------------------------------------------------------
# Nama kolom yang ditampilkan pada tabel, grafik, dan kartu skor
# ---------------------------------------------------------------------------
NAMA_VARIABEL = {
    "Marital_status": "Status pernikahan",
    "Application_mode": "Jalur pendaftaran",
    "Application_order": "Urutan pilihan prodi",
    "Course": "Program studi",
    "Daytime_evening_attendance": "Waktu perkuliahan",
    "Previous_qualification": "Ijazah terakhir",
    "Previous_qualification_grade": "Nilai ijazah terakhir",
    "Nacionality": "Kewarganegaraan",
    "Mothers_qualification": "Pendidikan ibu",
    "Fathers_qualification": "Pendidikan ayah",
    "Mothers_occupation": "Pekerjaan ibu",
    "Fathers_occupation": "Pekerjaan ayah",
    "Admission_grade": "Nilai seleksi masuk",
    "Displaced": "Tempat tinggal",
    "Educational_special_needs": "Kebutuhan khusus",
    "Debtor": "Riwayat tunggakan",
    "Tuition_fees_up_to_date": "Kelancaran uang kuliah",
    "Gender": "Jenis kelamin",
    "Scholarship_holder": "Beasiswa",
    "Age_at_enrollment": "Usia mendaftar",
    "International": "Status kewarganegaraan",
    "Curricular_units_1st_sem_credited": "MK alih kredit semester 1",
    "Curricular_units_1st_sem_enrolled": "MK diambil semester 1",
    "Curricular_units_1st_sem_evaluations": "Ujian diikuti semester 1",
    "Curricular_units_1st_sem_approved": "MK lulus semester 1",
    "Curricular_units_1st_sem_grade": "Nilai rata rata semester 1",
    "Curricular_units_1st_sem_without_evaluations": "MK tanpa ujian semester 1",
    "Curricular_units_2nd_sem_credited": "MK alih kredit semester 2",
    "Curricular_units_2nd_sem_enrolled": "MK diambil semester 2",
    "Curricular_units_2nd_sem_evaluations": "Ujian diikuti semester 2",
    "Curricular_units_2nd_sem_approved": "MK lulus semester 2",
    "Curricular_units_2nd_sem_grade": "Nilai rata rata semester 2",
    "Curricular_units_2nd_sem_without_evaluations": "MK tanpa ujian semester 2",
    "Unemployment_rate": "Angka pengangguran nasional",
    "Inflation_rate": "Angka inflasi nasional",
    "GDP": "Pertumbuhan ekonomi nasional",
}

# Satuan pendek untuk melengkapi label pita pada formulir prototipe.
SATUAN = {
    "Age_at_enrollment": "tahun",
    "Admission_grade": "poin",
    "Previous_qualification_grade": "poin",
    "Curricular_units_1st_sem_grade": "skala 0 sampai 20",
    "Curricular_units_2nd_sem_grade": "skala 0 sampai 20",
}


def nama_variabel(kolom: str) -> str:
    """Nama kolom dalam bahasa Indonesia, atau nama aslinya bila belum terdaftar."""
    return NAMA_VARIABEL.get(kolom, kolom)


def terjemahkan(kolom: str, nilai) -> str:
    """Isi sel dalam bahasa Indonesia.

    Nilai pita hasil pembagian kuantil dibiarkan apa adanya karena sudah berupa
    teks yang dapat dibaca, misalnya "22 sampai 28". Yang diterjemahkan hanyalah
    kode kategori dan penanda nol satu.
    """
    padanan = TERJEMAHAN.get(kolom)
    if padanan is None:
        return str(nilai)
    try:
        kunci = int(float(nilai))
    except (TypeError, ValueError):
        return str(nilai)
    return padanan.get(kunci, str(nilai))


def label_pita_kategori(kolom: str, pita: str) -> str:
    """Label pita untuk kolom kategori, termasuk pita gabungan kategori langka."""
    if pita == "Kategori lain":
        return "Kategori lain yang jarang muncul"
    return terjemahkan(kolom, pita)
