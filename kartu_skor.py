"""
kartu_skor.py
=============
Mesin kartu skor risiko dropout mahasiswa Jaya Jaya Institut.

Berkas ini memuat seluruh perkakas metode kartu skor, yaitu pendekatan yang lazim
dipakai lembaga keuangan untuk menilai risiko dan di sini diterapkan pada risiko
mahasiswa berhenti studi. Alur kerjanya tiga langkah:

1. **Membagi setiap variabel menjadi pita.** Variabel angka dibagi memakai kuantil,
   variabel kode dikelompokkan dan kategori langka digabung. Hasilnya bukan angka
   mentah lagi, melainkan pita yang dapat dibaca manusia, misalnya "usia 24 sampai
   27 tahun".
2. **Menghitung Weight of Evidence tiap pita.** WOE mengukur seberapa condong
   sebuah pita ke arah lulus atau ke arah berhenti dibanding populasi keseluruhan.
   Penjumlahan kontribusinya menghasilkan Information Value, ukuran kekuatan satu
   variabel secara utuh.
3. **Mengubah koefisien regresi menjadi poin.** Model regresi logistik dilatih di
   atas nilai WOE, lalu koefisiennya diterjemahkan menjadi poin bilangan bulat.
   Staf akademik cukup menjumlahkan poin tiap pita untuk memperoleh skor akhir,
   tanpa perlu menjalankan program apa pun.

Keputusan memakai metode ini diambil karena institusi membutuhkan alasan yang bisa
disampaikan kepada mahasiswa. Kalimat "skor Anda 512 karena tunggakan menyumbang
minus 48 poin" jauh lebih dapat ditindaklanjuti daripada "model memberi peluang
0,71".

Modul ini dipakai oleh notebook.ipynb saat pelatihan, oleh app.py saat penilaian,
dan oleh skrip di folder build saat menyiapkan data dashboard. Karena objek hasil
pelatihan disimpan dengan joblib, berkas ini wajib ikut ter-deploy bersama app.py.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------------
# Daftar kolom dataset
# ---------------------------------------------------------------------------

# Kolom kode kategori. Angka di dalamnya adalah penanda, bukan besaran, sehingga
# tidak boleh dibagi menjadi pita berdasarkan urutan nilainya.
KOLOM_KODE = [
    "Marital_status", "Application_mode", "Course", "Previous_qualification",
    "Nacionality", "Mothers_qualification", "Fathers_qualification",
    "Mothers_occupation", "Fathers_occupation",
]

# Kolom bernilai 0 atau 1. Keduanya sudah berupa pita, cukup dua kategori.
KOLOM_TANDA = [
    "Daytime_evening_attendance", "Displaced", "Educational_special_needs", "Debtor",
    "Tuition_fees_up_to_date", "Gender", "Scholarship_holder", "International",
]

# Kolom angka yang dibagi menjadi pita memakai kuantil.
KOLOM_ANGKA = [
    "Application_order", "Previous_qualification_grade", "Admission_grade",
    "Age_at_enrollment",
    "Curricular_units_1st_sem_credited", "Curricular_units_1st_sem_enrolled",
    "Curricular_units_1st_sem_evaluations", "Curricular_units_1st_sem_approved",
    "Curricular_units_1st_sem_grade", "Curricular_units_1st_sem_without_evaluations",
    "Curricular_units_2nd_sem_credited", "Curricular_units_2nd_sem_enrolled",
    "Curricular_units_2nd_sem_evaluations", "Curricular_units_2nd_sem_approved",
    "Curricular_units_2nd_sem_grade", "Curricular_units_2nd_sem_without_evaluations",
    "Unemployment_rate", "Inflation_rate", "GDP",
]

SELURUH_KOLOM = KOLOM_KODE + KOLOM_TANDA + KOLOM_ANGKA

# Empat sumbu yang dipakai untuk mengelompokkan mahasiswa berisiko menjadi
# beberapa profil. Pengelompokan sengaja tidak dilakukan di atas kolom mentah,
# melainkan di atas sumbu bentukan yang masing masing mewakili satu jenis masalah.
# Dengan begitu klaster yang terbentuk langsung dapat diberi nama dan diberi
# bentuk bantuan yang berbeda, bukan sekadar bernama klaster satu sampai tiga.
SUMBU_PROFIL = ["tekanan_biaya", "kesulitan_akademik", "usia_masuk", "kehadiran_ujian"]


def sumbu_profil(X: pd.DataFrame) -> pd.DataFrame:
    """Menghitung empat sumbu profil mahasiswa.

    * ``tekanan_biaya``      : semakin besar semakin berat beban keuangannya,
      menggabungkan tunggakan uang kuliah, catatan menunggak, dan beasiswa.
    * ``kesulitan_akademik`` : bagian mata kuliah tahun pertama yang tidak lulus.
    * ``usia_masuk``         : usia saat mendaftar, penanda tahap hidup.
    * ``kehadiran_ujian``    : rata rata ujian yang diikuti per mata kuliah yang
      diambil. Nilai rendah menandakan mahasiswa yang menghilang dari kelas,
      berbeda dari mahasiswa yang hadir tetapi tidak lulus.
    """
    diambil = (X["Curricular_units_1st_sem_enrolled"]
               + X["Curricular_units_2nd_sem_enrolled"]).replace(0, np.nan)
    lulus = (X["Curricular_units_1st_sem_approved"]
             + X["Curricular_units_2nd_sem_approved"])
    ujian = (X["Curricular_units_1st_sem_evaluations"]
             + X["Curricular_units_2nd_sem_evaluations"])
    return pd.DataFrame({
        "tekanan_biaya": (2 * (1 - X["Tuition_fees_up_to_date"]) + X["Debtor"]
                          - X["Scholarship_holder"]).astype(float),
        "kesulitan_akademik": (1 - (lulus / diambil)).fillna(1.0),
        "usia_masuk": X["Age_at_enrollment"].astype(float),
        "kehadiran_ujian": (ujian / diambil).fillna(0.0),
    }, index=X.index)


class PitaWOE(BaseEstimator, TransformerMixin):
    """Membagi variabel menjadi pita lalu menggantinya dengan nilai Weight of Evidence.

    Perjanjian tanda yang dipakai di seluruh proyek ini:

    * kelompok **aman** adalah mahasiswa yang lulus, ditandai y = 0,
    * kelompok **berisiko** adalah mahasiswa yang berhenti, ditandai y = 1,
    * ``WOE = ln(bagian aman / bagian berisiko)``.

    Dengan perjanjian itu, WOE positif berarti pita tersebut lebih banyak diisi
    mahasiswa yang selamat dibanding rata rata populasi, sehingga nantinya
    memperoleh poin kartu skor yang lebih tinggi. Arah ini dipilih supaya skor
    besar selalu berarti kabar baik, sama seperti pada kartu skor kredit.

    Parameter
    ---------
    jumlah_pita : int
        Banyak pita kuantil untuk variabel angka. Bila variabel memiliki nilai unik
        lebih sedikit daripada angka ini, jumlah pitanya menyesuaikan sendiri.
    bagian_minimum : float
        Kategori dengan porsi di bawah nilai ini digabung menjadi satu pita
        bernama "Kategori lain", supaya WOE tidak dihitung dari segelintir baris.
    penghalus : float
        Tambahan kecil pada penghitungan agar pita yang hanya berisi satu kelompok
        tidak menghasilkan pembagian dengan nol.
    """

    def __init__(self, jumlah_pita=5, bagian_minimum=0.03, penghalus=0.5, monoton=True,
                 toleransi_arah=0.01):
        self.jumlah_pita = jumlah_pita
        self.bagian_minimum = bagian_minimum
        self.penghalus = penghalus
        self.monoton = monoton
        self.toleransi_arah = toleransi_arah

    # -- pembentukan pita ---------------------------------------------------
    def _batas_kuantil(self, deret: pd.Series):
        """Menyusun batas pita kuantil, digabung bila ada batas yang berulang."""
        kuantil = np.linspace(0, 1, self.jumlah_pita + 1)
        # Batas dipaksa bertipe pecahan karena ujungnya diganti tak hingga, dan
        # sebagian kolom dataset ini bertipe bilangan bulat.
        batas = np.unique(np.quantile(deret.dropna(), kuantil)).astype(float)
        if len(batas) < 3:
            batas = np.unique(deret.dropna()).astype(float)
            if len(batas) < 2:
                return None
        batas[0] = -np.inf
        batas[-1] = np.inf
        return batas

    @staticmethod
    def _label_pita(batas, bilangan_bulat=False):
        """Menulis nama pita yang enak dibaca.

        Pita selalu terbuka di kiri dan tertutup di kanan. Untuk kolom bilangan
        bulat, penamaan memakai nilai yang benar benar mungkin muncul, sehingga
        pita (18, 19] ditulis "19" dan bukan "18 sampai 19". Tanpa penyesuaian ini
        kartu skor jadi membingungkan, karena angka 18 akan tampak berada di dua
        pita sekaligus.
        """
        nama = []
        for kiri, kanan in zip(batas[:-1], batas[1:]):
            if bilangan_bulat:
                mulai = int(np.floor(kiri)) + 1 if np.isfinite(kiri) else None
                akhir = int(np.floor(kanan)) if np.isfinite(kanan) else None
                if mulai is None:
                    nama.append(f"{akhir} ke bawah")
                elif akhir is None:
                    nama.append(f"{mulai} ke atas")
                elif mulai >= akhir:
                    nama.append(f"{akhir}")
                else:
                    nama.append(f"{mulai} sampai {akhir}")
                continue
            if np.isneginf(kiri):
                nama.append(f"sampai {kanan:g}".replace(".", ","))
            elif np.isposinf(kanan):
                nama.append(f"lebih dari {kiri:g}".replace(".", ","))
            else:
                nama.append(f"{kiri:g} sampai {kanan:g}".replace(".", ","))
        return nama

    def _rapikan_monoton(self, deret: pd.Series, batas, y) -> np.ndarray:
        """Menggabungkan pita bertetangga sampai tingkat dropoutnya bergerak satu arah.

        Kartu skor yang baik tidak boleh memuat lompatan yang berlawanan arah,
        misalnya pita "7 mata kuliah lulus" memperoleh poin lebih tinggi daripada
        pita "8 ke atas". Lompatan seperti itu biasanya hanya riak sampel, bukan
        pola nyata, dan membuat staf sulit menjelaskan kartu kepada mahasiswa.

        Selama masih ada pelanggaran arah, sepasang pita bertetangga dengan selisih
        tingkat dropout terkecil digabungkan, lalu pemeriksaan diulang. Proses
        berhenti ketika arahnya konsisten atau ketika hanya tersisa dua pita.
        """
        batas = list(batas)
        nilai = deret.to_numpy()
        y = np.asarray(y).astype(int)
        while len(batas) > 3:
            indeks = np.digitize(nilai, batas[1:-1], right=True)
            tingkat = np.array([y[indeks == i].mean() if (indeks == i).any() else np.nan
                                for i in range(len(batas) - 1)])
            if np.isnan(tingkat).any():
                # Pita kosong langsung digabung dengan tetangga di sebelah kirinya.
                kosong = int(np.where(np.isnan(tingkat))[0][0])
                batas.pop(max(1, min(kosong, len(batas) - 2)))
                continue
            arah = np.sign(tingkat[-1] - tingkat[0]) or 1.0
            selisih = np.diff(tingkat) * arah
            # Pelanggaran kecil dibiarkan. Tanpa toleransi, riak sebesar setengah
            # poin persen pun memicu penggabungan, dan variabel terkuat berakhir
            # hanya memiliki dua pita sehingga kartu kehilangan ketajamannya.
            if (selisih >= -self.toleransi_arah).all():
                break
            pelanggar = int(np.argmin(selisih))
            batas.pop(pelanggar + 1)
        return np.array(batas, dtype=float)

    def _pita_satu_kolom(self, deret: pd.Series, nama_kolom: str) -> pd.Series:
        """Mengubah satu kolom menjadi label pita, memakai aturan yang sudah dipelajari."""
        aturan = self.aturan_[nama_kolom]
        if aturan["jenis"] == "angka":
            hasil = pd.cut(deret, aturan["batas"], labels=aturan["label"],
                           include_lowest=True)
            return hasil.astype(str)
        nilai = deret.astype(str)
        return nilai.where(nilai.isin(aturan["kategori"]), "Kategori lain")

    # -- pelatihan ----------------------------------------------------------
    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y).astype(int)
        self.kolom_ = [k for k in SELURUH_KOLOM if k in X.columns]
        self.aturan_ = {}
        self.woe_ = {}
        self.iv_ = {}
        self.ringkasan_ = []

        jumlah_aman = int((y == 0).sum())
        jumlah_berisiko = int((y == 1).sum())

        for kolom in self.kolom_:
            deret = X[kolom]
            if kolom in KOLOM_ANGKA:
                batas = self._batas_kuantil(deret)
                if batas is None:
                    continue
                if self.monoton:
                    batas = self._rapikan_monoton(deret, batas, y)
                bulat = bool(np.all(np.mod(deret.dropna().to_numpy(), 1) == 0))
                self.aturan_[kolom] = {"jenis": "angka", "batas": batas,
                                       "label": self._label_pita(batas, bulat)}
            else:
                porsi = deret.astype(str).value_counts(normalize=True)
                simpan = list(porsi[porsi >= self.bagian_minimum].index)
                if not simpan:
                    simpan = list(porsi.head(3).index)
                self.aturan_[kolom] = {"jenis": "kode", "kategori": simpan}

            pita = self._pita_satu_kolom(deret, kolom)
            tabel = pd.DataFrame({"pita": pita, "y": y})
            rekap = tabel.groupby("pita", observed=True).agg(
                jumlah=("y", "size"), berisiko=("y", "sum"))
            rekap["aman"] = rekap.jumlah - rekap.berisiko

            bagian_aman = (rekap.aman + self.penghalus) / (jumlah_aman + self.penghalus)
            bagian_berisiko = ((rekap.berisiko + self.penghalus)
                               / (jumlah_berisiko + self.penghalus))
            rekap["woe"] = np.log(bagian_aman / bagian_berisiko)
            rekap["sumbangan_iv"] = (bagian_aman - bagian_berisiko) * rekap.woe
            rekap["tingkat_dropout"] = rekap.berisiko / rekap.jumlah

            self.woe_[kolom] = rekap.woe.to_dict()
            self.iv_[kolom] = float(rekap.sumbangan_iv.sum())
            for pita_nama, baris in rekap.iterrows():
                self.ringkasan_.append({
                    "kolom": kolom,
                    "pita": pita_nama,
                    "jumlah": int(baris.jumlah),
                    "porsi": baris.jumlah / len(X),
                    "tingkat_dropout": float(baris.tingkat_dropout),
                    "woe": float(baris.woe),
                    "sumbangan_iv": float(baris.sumbangan_iv),
                })
        return self

    # -- penerapan ----------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        hasil = {}
        for kolom in self.kolom_:
            if kolom not in self.aturan_:
                continue
            pita = self._pita_satu_kolom(X[kolom], kolom)
            # Pita yang belum pernah terlihat diberi WOE nol, artinya dianggap
            # sama persis dengan rata rata populasi dan tidak menggeser skor.
            hasil[f"woe_{kolom}"] = pita.map(self.woe_[kolom]).fillna(0.0).to_numpy()
        return pd.DataFrame(hasil, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"woe_{k}" for k in self.kolom_ if k in self.aturan_])

    # -- laporan ------------------------------------------------------------
    def tabel_pita(self) -> pd.DataFrame:
        """Tabel rinci seluruh pita beserta WOE dan sumbangan Information Value."""
        return pd.DataFrame(self.ringkasan_)

    def tabel_iv(self) -> pd.DataFrame:
        """Peringkat variabel menurut Information Value beserta tafsirannya."""
        tabel = (pd.DataFrame({"kolom": list(self.iv_), "iv": list(self.iv_.values())})
                 .sort_values("iv", ascending=False)
                 .reset_index(drop=True))
        tabel["kekuatan"] = pd.cut(
            tabel.iv, [-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf],
            labels=["tidak berguna", "lemah", "sedang", "kuat", "sangat kuat"])
        return tabel


def susun_kartu(pembagi: PitaWOE, model, kolom_terpakai, pdo=20.0,
                skor_dasar=600.0, odds_dasar=1.0) -> pd.DataFrame:
    """Menerjemahkan koefisien regresi logistik menjadi tabel poin kartu skor.

    Rumus baku kartu skor dipakai apa adanya:

        faktor = pdo / ln(2)
        offset = skor_dasar - faktor * ln(odds_dasar)
        poin_pita = -(koefisien * woe + intersep / n) * faktor + offset / n

    dengan n banyaknya variabel yang masuk kartu. Karena setiap variabel menyumbang
    satu suku, skor akhir seorang mahasiswa cukup dihitung dengan menjumlahkan poin
    seluruh pitanya. Nilai pdo 20 berarti setiap tambahan 20 poin melipatduakan
    perbandingan peluang selamat terhadap peluang berhenti.

    Parameter ``kolom_terpakai`` berisi nama kolom asli yang dipakai model, dengan
    urutan yang sama seperti urutan koefisien.
    """
    faktor = pdo / np.log(2)
    offset = skor_dasar - faktor * np.log(odds_dasar)
    koefisien = dict(zip(kolom_terpakai, model.coef_[0]))
    intersep = float(model.intercept_[0])
    jumlah_kolom = len(kolom_terpakai)

    baris = []
    for kolom in kolom_terpakai:
        for pita, woe in pembagi.woe_[kolom].items():
            poin = -(koefisien[kolom] * woe + intersep / jumlah_kolom) * faktor \
                + offset / jumlah_kolom
            baris.append({"kolom": kolom, "pita": pita, "woe": float(woe),
                          "poin": int(round(poin))})
    kartu = pd.DataFrame(baris)
    kartu["poin_dasar"] = kartu.groupby("kolom").poin.transform("mean").round().astype(int)
    kartu["selisih_poin"] = kartu.poin - kartu.poin_dasar
    return kartu


def hitung_skor(kartu: pd.DataFrame, pembagi: PitaWOE, X: pd.DataFrame) -> np.ndarray:
    """Menjumlahkan poin kartu skor untuk setiap baris mahasiswa."""
    peta = {(baris.kolom, baris.pita): baris.poin for baris in kartu.itertuples()}
    kolom_kartu = list(dict.fromkeys(kartu.kolom))
    total = np.zeros(len(X), dtype=float)
    rata_kolom = kartu.groupby("kolom").poin.mean().to_dict()
    for kolom in kolom_kartu:
        pita = pembagi._pita_satu_kolom(X[kolom], kolom)
        poin = np.array([peta.get((kolom, p), rata_kolom[kolom]) for p in pita])
        total += poin
    return total


def rincian_skor(kartu: pd.DataFrame, pembagi: PitaWOE, baris: pd.DataFrame) -> pd.DataFrame:
    """Menjelaskan skor satu mahasiswa: pita apa yang dipakai dan berapa poinnya.

    Keluarannya diurutkan dari pengurang poin terbesar, karena itulah yang perlu
    dibicarakan lebih dulu ketika dosen wali menemui mahasiswa bersangkutan.
    """
    peta = {(b.kolom, b.pita): (b.poin, b.selisih_poin) for b in kartu.itertuples()}
    hasil = []
    for kolom in dict.fromkeys(kartu.kolom):
        pita = pembagi._pita_satu_kolom(baris[kolom], kolom).iloc[0]
        poin, selisih = peta.get((kolom, pita), (0, 0))
        hasil.append({"kolom": kolom, "pita": pita, "poin": poin, "selisih": selisih})
    return pd.DataFrame(hasil).sort_values("selisih")


PERINGKAT = ["A", "B", "C", "D", "E"]


def beri_peringkat(skor, batas) -> pd.Series:
    """Memberi peringkat huruf A sampai E berdasarkan empat batas skor.

    Peringkat A adalah kelompok paling aman. Batas diberikan dari yang tertinggi ke
    yang terendah, misalnya [640, 600, 560, 520].
    """
    nilai = np.asarray(skor, dtype=float)
    hasil = np.full(len(nilai), PERINGKAT[-1], dtype=object)
    # Batas ditelusuri dari yang tertinggi, dan setiap mahasiswa hanya boleh
    # menerima peringkat pertama yang memenuhi supaya tidak tertimpa batas berikutnya.
    for huruf, ambang in zip(PERINGKAT[:-1], batas):
        belum_terisi = hasil == PERINGKAT[-1]
        hasil[belum_terisi & (nilai >= ambang)] = huruf
    return pd.Series(hasil, index=getattr(skor, "index", None))
