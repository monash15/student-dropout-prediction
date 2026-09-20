"""
app.py
======
Kalkulator kartu skor risiko dropout mahasiswa Jaya Jaya Institut.

Prototipe ini adalah bentuk digital dari kartu skor yang dihasilkan notebook.ipynb.
Cara kerjanya sengaja dibuat sama persis dengan kartu cetak: staf memilih pita yang
sesuai untuk setiap variabel, poinnya dijumlahkan, lalu totalnya diterjemahkan
menjadi peringkat huruf dan peluang. Tidak ada perhitungan tersembunyi, dan seluruh
angka yang muncul dapat ditelusuri sampai barisnya.

Empat ruang kerja yang tersedia:

1. Hitung kartu skor  : menilai satu mahasiswa lewat pilihan pita.
2. Antrean prioritas  : menilai satu berkas CSV lalu menyusun urutan bimbingan.
3. Kartu skor lengkap : seluruh tabel poin, siap dicetak.
4. Cara kerja         : ringkasan metode, ukuran kemampuan, dan batasannya.

Menjalankan secara lokal:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import kartu_skor as ksk
import label_pita as lb

DASAR = Path(__file__).parent

st.set_page_config(
    page_title="Kartu Skor Risiko Dropout | Jaya Jaya Institut",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palet sama dengan notebook supaya laporan, kartu cetak, dan aplikasi terbaca
# sebagai satu kesatuan. Penanda status memakai warna dan garis, tanpa ikon.
NILA = "#4338ca"
BATA = "#c2410c"
BATU = "#57534e"
GARIS = "#e7e5e4"
TINTA = "#1c1917"
LATAR = "#fafaf9"
WARNA_PERINGKAT = {"A": "#166534", "B": "#4d7c0f", "C": "#a16207",
                   "D": "#b45309", "E": "#9f1239"}

st.markdown(
    f"""
    <style>
        .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }}
        h1, h2, h3 {{ color: {TINTA}; letter-spacing: -0.01em; }}
        .judul {{ border-left: 5px solid {NILA}; padding-left: 0.9rem; margin-bottom: 1.2rem; }}
        .judul h1 {{ font-size: 1.6rem; margin: 0 0 0.2rem 0; }}
        .judul p {{ color: {BATU}; margin: 0; font-size: 0.92rem; }}
        .panel {{
            background: {LATAR}; border: 1px solid {GARIS}; border-radius: 8px;
            padding: 1rem 1.1rem; margin-bottom: 0.7rem;
        }}
        .angka {{ font-size: 2.2rem; font-weight: 700; line-height: 1.1; }}
        .keterangan {{ font-size: 0.8rem; color: {BATU}; text-transform: uppercase;
                       letter-spacing: 0.04em; }}
        .baris-poin {{
            display: flex; justify-content: space-between; padding: 0.45rem 0.7rem;
            border-bottom: 1px solid {GARIS}; font-size: 0.9rem;
        }}
        .tambah {{ border-left: 4px solid #166534; }}
        .kurang {{ border-left: 4px solid #9f1239; }}
        .catatan {{ font-size: 0.85rem; color: {BATU}; border-top: 1px solid {GARIS};
                    padding-top: 0.7rem; margin-top: 1.1rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Memuat kartu skor")
def muat_mesin():
    """Memuat mesin kartu skor beserta metadatanya satu kali per sesi."""
    mesin = joblib.load(DASAR / "model" / "mesin_kartu_skor.joblib")
    with open(DASAR / "model" / "metadata_kartu.json", encoding="utf-8") as berkas:
        info = json.load(berkas)
    return mesin, info


@st.cache_data(show_spinner=False)
def muat_contoh():
    jalur = DASAR / "dataset" / "mahasiswa_aktif.csv"
    return pd.read_csv(jalur, sep=";") if jalur.exists() else None


MESIN, INFO = muat_mesin()
KARTU = MESIN["kartu"]
PEMBAGI = MESIN["pembagi"]
VARIABEL = MESIN["variabel_kartu"]
BATAS = MESIN["batas_peringkat"]
POTONG = MESIN["skor_potong"]
PROFIL = MESIN["profil"]
TINDAKAN = MESIN["tindakan"]
PDO = INFO["kartu"]["pdo"]
SKOR_DASAR = INFO["kartu"]["skor_dasar"]


def peluang_dari_skor(skor: float) -> float:
    """Menerjemahkan skor menjadi peluang berhenti studi.

    Kartu disusun dengan skor dasar 600 pada odds satu banding satu dan PDO 20,
    sehingga hubungannya tetap: odds selamat = 2 pangkat ((skor - 600) / 20).
    """
    odds = 2.0 ** ((skor - SKOR_DASAR) / PDO)
    return float(1.0 / (1.0 + odds))


def label_baris(baris) -> str:
    if baris.kolom in lb.TERJEMAHAN:
        return lb.label_pita_kategori(baris.kolom, baris.pita)
    return str(baris.pita)


PILIHAN_PITA = {}
for kolom in VARIABEL:
    bagian = KARTU[KARTU.kolom == kolom].sort_values("poin", ascending=False)
    PILIHAN_PITA[kolom] = [(label_baris(b), int(b.poin), int(b.selisih_poin), b.pita)
                           for b in bagian.itertuples()]

# ---------------------------------------------------------------------------
# Bilah sisi
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Jaya Jaya Institut")
    st.caption("Kartu skor risiko dropout mahasiswa")
    st.markdown("---")
    st.markdown("**Ringkasan kartu**")
    kiri, kanan = st.columns(2)
    kiri.metric("Variabel", len(VARIABEL))
    kanan.metric("Baris pita", INFO["kartu"]["jumlah_baris"])
    kiri.metric("Skor potong", int(POTONG))
    kanan.metric("ROC-AUC", f"{INFO['metrik_uji']['ROC-AUC']:.3f}")
    st.caption(
        f"Statistik KS {INFO['metrik_uji']['KS data latih']:.3f}. Skor dasar "
        f"{SKOR_DASAR} pada peluang seimbang, setiap {PDO} poin melipatduakan "
        "perbandingan peluang selamat."
    )
    st.markdown("---")
    st.markdown("**Peringkat huruf**")
    keterangan_peringkat = [
        ("A", f"{BATAS[0]} ke atas", "aman"),
        ("B", f"{BATAS[1]} sampai {BATAS[0] - 1}", "aman"),
        ("C", f"{BATAS[2]} sampai {BATAS[1] - 1}", "pantau"),
        ("D", f"{BATAS[3]} sampai {BATAS[2] - 1}", "dipanggil"),
        ("E", f"di bawah {BATAS[3]}", "dipanggil"),
    ]
    for huruf, rentang, status in keterangan_peringkat:
        st.markdown(
            f"<div style='border-left:4px solid {WARNA_PERINGKAT[huruf]};"
            f"padding:0.15rem 0 0.15rem 0.6rem;margin-bottom:0.25rem;font-size:0.85rem;'>"
            f"<strong>{huruf}</strong>, skor {rentang}, {status}</div>",
            unsafe_allow_html=True)
    st.markdown("---")
    st.caption(
        f"Disusun oleh {INFO['identitas']['nama']} untuk proyek akhir kelas "
        "Belajar Penerapan Data Science, Dicoding."
    )

st.markdown(
    """
    <div class="judul">
        <h1>Kalkulator Kartu Skor Risiko Dropout</h1>
        <p>Menilai risiko mahasiswa berhenti studi dengan cara yang dapat dijelaskan
        kembali kepada mahasiswanya, satu baris poin pada satu waktu.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_hitung, tab_antrean, tab_kartu, tab_cara = st.tabs(
    ["Hitung kartu skor", "Antrean prioritas", "Kartu skor lengkap", "Cara kerja"]
)


def panel_angka(kolom, keterangan, nilai, warna=TINTA):
    kolom.markdown(
        f"""
        <div class="panel">
            <div class="keterangan">{keterangan}</div>
            <div class="angka" style="color:{warna}">{nilai}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tab 1: menghitung skor satu mahasiswa
# ---------------------------------------------------------------------------
with tab_hitung:
    st.markdown("#### Menilai satu mahasiswa")
    st.caption(
        "Pilih pita yang sesuai untuk setiap variabel. Angka di dalam kurung adalah poin "
        "yang disumbangkan pita tersebut, persis seperti pada kartu cetak."
    )

    with st.form("kartu_mahasiswa"):
        terpilih = {}
        kolom_bentuk = st.columns(2)
        for nomor, kolom in enumerate(VARIABEL):
            wadah = kolom_bentuk[nomor % 2]
            daftar = PILIHAN_PITA[kolom]
            terpilih[kolom] = wadah.selectbox(
                lb.nama_variabel(kolom),
                options=list(range(len(daftar))),
                format_func=lambda i, d=daftar: f"{d[i][0]}   ({d[i][1]} poin)",
                key=f"pita_{kolom}",
            )

        st.markdown("**Data tambahan untuk menentukan profil**")
        st.caption(
            "Bagian ini tidak menambah maupun mengurangi satu poin pun. Isinya hanya "
            "dipakai untuk menebak jenis masalah yang dihadapi mahasiswa, sehingga "
            "bentuk bantuannya dapat dibedakan."
        )
        tambahan = st.columns(4)
        mk_diambil_1 = tambahan[0].number_input("MK diambil semester 1", 0, 30, 6)
        mk_diambil_2 = tambahan[1].number_input("MK diambil semester 2", 0, 30, 6)
        ujian_1 = tambahan[2].number_input("Ujian diikuti semester 1", 0, 45, 8)
        ujian_2 = tambahan[3].number_input("Ujian diikuti semester 2", 0, 45, 8)

        hitung = st.form_submit_button("Hitung skor mahasiswa", type="primary")

    if hitung:
        rincian = []
        total = 0
        for kolom in VARIABEL:
            label, poin, selisih, pita_asli = PILIHAN_PITA[kolom][terpilih[kolom]]
            total += poin
            rincian.append({"Variabel": lb.nama_variabel(kolom), "Pita terpilih": label,
                            "Poin": poin, "Selisih": selisih, "kolom": kolom,
                            "pita": pita_asli})
        rincian = pd.DataFrame(rincian)
        peringkat = str(ksk.beri_peringkat(np.array([total]), BATAS).iloc[0])
        peluang = peluang_dari_skor(total)

        st.markdown("---")
        panel = st.columns(4)
        panel_angka(panel[0], "Skor kartu", f"{total}", NILA)
        panel_angka(panel[1], "Peringkat", peringkat, WARNA_PERINGKAT[peringkat])
        panel_angka(panel[2], "Peluang berhenti studi", f"{peluang * 100:.1f}%",
                    WARNA_PERINGKAT[peringkat])
        panel_angka(panel[3], "Skor potong", f"{int(POTONG)}")

        posisi = min(max((total - 380) / (760 - 380), 0), 1) * 100
        letak_potong = (POTONG - 380) / (760 - 380) * 100
        st.markdown(
            f"""
            <div style="margin:0.2rem 0 1.2rem 0;">
              <div style="position:relative;height:15px;border-radius:8px;
                          background:linear-gradient(90deg,#9f1239 0%,#b45309 22%,
                          #a16207 42%,#4d7c0f 66%,#166534 100%);">
                <div style="position:absolute;left:{posisi}%;top:-6px;width:3px;height:27px;
                            background:{TINTA};"></div>
                <div style="position:absolute;left:{letak_potong}%;top:-3px;width:2px;
                            height:21px;background:#ffffff;"></div>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:0.75rem;
                          color:{BATU};margin-top:0.3rem;">
                <span>380</span><span>skor potong {int(POTONG)}</span><span>760</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        kolom_kiri, kolom_kanan = st.columns([1.3, 1])
        with kolom_kiri:
            st.markdown("**Rincian poin, diurutkan dari yang paling menekan skor**")
            for baris in rincian.sort_values("Selisih").itertuples():
                arah = "kurang" if baris.Selisih < 0 else "tambah"
                st.markdown(
                    f"""
                    <div class="baris-poin {arah}">
                        <span><strong>{baris.Variabel}</strong>: {baris._2}</span>
                        <span>{baris.Poin} poin ({baris.Selisih:+d})</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.caption(
                "Kolom selisih menunjukkan jarak poin pita ini terhadap poin dasar "
                "variabelnya. Nilai negatif berarti mahasiswa berada di bawah rata rata "
                "pada variabel tersebut."
            )

        with kolom_kanan:
            if peringkat in ("D", "E"):
                diambil = mk_diambil_1 + mk_diambil_2
                lulus_sem = {"Curricular_units_1st_sem_approved": 0,
                             "Curricular_units_2nd_sem_approved": 0}
                for kolom in lulus_sem:
                    if kolom in VARIABEL:
                        label = PILIHAN_PITA[kolom][terpilih[kolom]][0]
                        angka = [int(t) for t in str(label).split() if t.isdigit()]
                        lulus_sem[kolom] = angka[0] if angka else 0
                baris_profil = pd.DataFrame([{
                    "Curricular_units_1st_sem_enrolled": mk_diambil_1,
                    "Curricular_units_2nd_sem_enrolled": mk_diambil_2,
                    "Curricular_units_1st_sem_approved":
                        lulus_sem["Curricular_units_1st_sem_approved"],
                    "Curricular_units_2nd_sem_approved":
                        lulus_sem["Curricular_units_2nd_sem_approved"],
                    "Curricular_units_1st_sem_evaluations": ujian_1,
                    "Curricular_units_2nd_sem_evaluations": ujian_2,
                    "Tuition_fees_up_to_date": int(
                        "lancar" in str(PILIHAN_PITA["Tuition_fees_up_to_date"][
                            terpilih["Tuition_fees_up_to_date"]][0]).lower()),
                    "Debtor": int("tidak" not in str(PILIHAN_PITA["Debtor"][
                        terpilih["Debtor"]][0]).lower()),
                    "Scholarship_holder": int("bukan" not in str(
                        PILIHAN_PITA["Scholarship_holder"][
                            terpilih["Scholarship_holder"]][0]).lower()),
                    "Age_at_enrollment": 21,
                }])
                sumbu = ksk.sumbu_profil(baris_profil)
                nomor_profil = int(MESIN["pengelompok"].predict(
                    MESIN["penskala_profil"].transform(sumbu))[0])
                nama_profil = PROFIL[nomor_profil]["nama"]
                st.markdown(f"**Profil kemungkinan: {nama_profil}**")
                st.caption(PROFIL[nomor_profil]["keterangan"])
                st.markdown("**Tindak lanjut yang disarankan**")
                for tindakan in TINDAKAN[nama_profil]:
                    st.markdown(f"- {tindakan}")
            else:
                st.markdown("**Tindak lanjut yang disarankan**")
                st.markdown(
                    "- Tidak perlu masuk antrean bimbingan pada periode ini.\n"
                    "- Cukup dipantau lewat laporan rutin dosen wali tiap akhir semester.\n"
                    "- Periksa ulang bila status pembayaran atau hasil semester berubah."
                )
            st.caption(
                "Skor adalah urutan perhatian, bukan vonis. Keputusan akhir tetap pada "
                "dosen wali yang mengenal mahasiswanya."
            )


# ---------------------------------------------------------------------------
# Tab 2: antrean prioritas
# ---------------------------------------------------------------------------
with tab_antrean:
    st.markdown("#### Menyusun antrean prioritas dari satu berkas")
    st.caption(
        "Unggah berkas CSV berisi kolom asli dataset, atau pakai berkas contoh berisi "
        "mahasiswa yang masih aktif."
    )

    kolom_unggah, kolom_atur = st.columns([1.4, 1])
    berkas = kolom_unggah.file_uploader("Berkas CSV mahasiswa", type=["csv"])
    pakai_contoh = kolom_unggah.checkbox("Pakai berkas contoh mahasiswa aktif", value=True)

    data = None
    if berkas is not None:
        isi = berkas.getvalue().decode("utf-8", "replace")
        pemisah = ";" if isi.count(";") > isi.count(",") else ","
        data = pd.read_csv(io.StringIO(isi), sep=pemisah)
    elif pakai_contoh:
        data = muat_contoh()

    if data is None:
        st.info("Belum ada data. Unggah berkas CSV atau centang pemakaian berkas contoh.")
    else:
        kurang = [k for k in VARIABEL if k not in data.columns]
        if kurang:
            st.error("Berkas kekurangan kolom berikut: "
                     + ", ".join(lb.nama_variabel(k) for k in kurang))
        else:
            kapasitas = kolom_atur.slider(
                "Kapasitas bimbingan periode ini (jumlah mahasiswa)",
                min_value=5, max_value=min(400, len(data)),
                value=min(50, len(data)), step=5)
            kolom_atur.caption(
                "Antrean disusun dari skor terendah. Isi sesuai jumlah slot konselor "
                "yang benar benar tersedia."
            )

            skor = ksk.hitung_skor(KARTU, PEMBAGI, data)
            peringkat = ksk.beri_peringkat(skor, BATAS)
            hasil = pd.DataFrame({
                "ID": [f"MHS{n:04d}" for n in range(1, len(data) + 1)],
                "Program studi": data.Course.map(lb.TERJEMAHAN["Course"]).to_numpy(),
                "Usia": data.Age_at_enrollment.to_numpy(),
                "Uang kuliah": data.Tuition_fees_up_to_date.map(
                    lb.TERJEMAHAN["Tuition_fees_up_to_date"]).to_numpy(),
                "MK lulus tahun 1": (data.Curricular_units_1st_sem_approved
                                     + data.Curricular_units_2nd_sem_approved).to_numpy(),
                "Skor kartu": skor.round(0).astype(int),
                "Peringkat": peringkat.to_numpy(),
                "Peluang berhenti": [peluang_dari_skor(s) for s in skor],
            }).sort_values("Skor kartu").reset_index(drop=True)
            hasil.insert(0, "Antrean", np.arange(1, len(hasil) + 1))

            sumbu = ksk.sumbu_profil(data)
            nomor_profil = MESIN["pengelompok"].predict(
                MESIN["penskala_profil"].transform(sumbu))
            profil_nama = np.array([PROFIL[int(k)]["nama"] for k in nomor_profil])
            urutan_asli = np.argsort(skor)
            hasil["Profil"] = np.where(
                np.isin(hasil.Peringkat, ["D", "E"]), profil_nama[urutan_asli],
                "Tidak dipanggil")

            dipanggil = int(hasil.Peringkat.isin(["D", "E"]).sum())
            panel = st.columns(4)
            panel_angka(panel[0], "Mahasiswa dinilai", f"{len(hasil):,}")
            panel_angka(panel[1], "Peringkat D dan E", f"{dipanggil:,}", BATA)
            panel_angka(panel[2], "Terjadwal periode ini", f"{kapasitas:,}", NILA)
            panel_angka(panel[3], "Skor terendah", f"{int(hasil['Skor kartu'].min())}",
                        WARNA_PERINGKAT["E"])

            if dipanggil > kapasitas:
                st.markdown(
                    f"""
                    <div class="panel" style="border-left:4px solid {BATA};">
                    Sebanyak {dipanggil:,} mahasiswa berada di peringkat D dan E, sementara
                    kapasitas periode ini {kapasitas:,} orang. Antrean di bawah sudah
                    diurutkan dari skor terendah, sehingga slot yang tersedia terpakai untuk
                    mahasiswa yang paling membutuhkan, dan sisanya masuk daftar tunggu.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.dataframe(
                hasil.head(kapasitas), width="stretch", hide_index=True,
                column_config={
                    "Skor kartu": st.column_config.ProgressColumn(
                        "Skor kartu", min_value=380, max_value=780, format="%d"),
                    "Peluang berhenti": st.column_config.NumberColumn(
                        "Peluang berhenti", format="%.2f"),
                })

            st.download_button(
                "Unduh seluruh hasil penilaian sebagai CSV",
                data=hasil.to_csv(index=False, sep=";").encode("utf-8"),
                file_name="antrean_prioritas_kartu_skor.csv",
                mime="text/csv",
            )

            kolom_a, kolom_b = st.columns(2)
            with kolom_a:
                st.markdown("**Sebaran peringkat**")
                st.bar_chart(hasil.Peringkat.value_counts().reindex(ksk.PERINGKAT).fillna(0),
                             color=NILA, height=240)
            with kolom_b:
                st.markdown("**Profil mahasiswa yang dipanggil**")
                sebaran_profil = (hasil[hasil.Peringkat.isin(["D", "E"])]
                                  .Profil.value_counts())
                if len(sebaran_profil):
                    st.bar_chart(sebaran_profil, color=BATA, height=240)
                else:
                    st.info("Tidak ada mahasiswa berperingkat D atau E pada berkas ini.")


# ---------------------------------------------------------------------------
# Tab 3: kartu skor lengkap
# ---------------------------------------------------------------------------
with tab_kartu:
    st.markdown("#### Kartu skor lengkap")
    st.markdown(
        f"Kartu ini memuat **{len(VARIABEL)} variabel** dan "
        f"**{INFO['kartu']['jumlah_baris']} pita**. Skor seorang mahasiswa adalah "
        "penjumlahan poin seluruh pita yang ditempatinya, dan tidak ada perhitungan lain "
        "di belakangnya. Tabel ini dapat dicetak lalu dipakai tanpa komputer."
    )

    tampil = KARTU.copy()
    tampil["Variabel"] = tampil.kolom.map(lb.nama_variabel)
    tampil["Pita"] = [label_baris(b) for b in tampil.itertuples()]
    tampil = tampil[["Variabel", "Pita", "poin", "poin_dasar", "selisih_poin", "woe"]]
    tampil.columns = ["Variabel", "Pita", "Poin", "Poin dasar", "Selisih", "WOE"]
    st.dataframe(tampil.round({"WOE": 3}), width="stretch", hide_index=True, height=520)

    st.download_button(
        "Unduh kartu skor sebagai CSV",
        data=tampil.to_csv(index=False, sep=";").encode("utf-8"),
        file_name="kartu_skor_dropout.csv",
        mime="text/csv",
    )

    st.markdown("**Cara membaca**")
    st.markdown(
        f"""
        - **Poin** adalah nilai yang dijumlahkan untuk mendapatkan skor akhir.
        - **Poin dasar** adalah rata rata poin variabel tersebut, dipakai sebagai titik
          nol pembanding.
        - **Selisih** menunjukkan seberapa jauh sebuah pita menolong atau menghukum
          dibanding pita lain pada variabel yang sama.
        - **WOE** adalah bahan mentah perhitungannya, dicantumkan agar hasilnya dapat
          diperiksa ulang.
        - Total skor {SKOR_DASAR} berarti peluang selamat dan peluang berhenti sama besar.
          Setiap tambahan {PDO} poin melipatduakan perbandingan peluang selamat.
        """
    )


# ---------------------------------------------------------------------------
# Tab 4: cara kerja dan batasan
# ---------------------------------------------------------------------------
with tab_cara:
    st.markdown("#### Bagaimana kartu ini dibangun")
    st.markdown(
        """
        1. **Setiap variabel dibagi menjadi pita.** Variabel angka dibagi memakai kuantil,
           lalu pita bertetangga yang tingkat dropoutnya berbalik arah digabung sampai
           seluruhnya bergerak satu arah. Variabel kode dikelompokkan, dan kategori yang
           terlalu jarang disatukan menjadi satu pita.
        2. **Tiap pita diukur dengan Weight of Evidence.** Penjumlahan sumbangannya
           menghasilkan Information Value, ukuran kekuatan satu variabel secara utuh.
        3. **Variabel disaring tiga kali:** yang terlalu lemah dibuang, yang tidak patut
           dipakai dibuang, lalu diambil dua belas teratas.
        4. **Regresi logistik dilatih di atas nilai WOE,** dan variabel yang koefisiennya
           berlawanan arah dibuang satu per satu sampai seluruh arah konsisten.
        5. **Koefisien diterjemahkan menjadi poin** memakai rumus baku kartu skor.
        """
    )

    kolom_a, kolom_b = st.columns(2)
    with kolom_a:
        st.markdown("**Kemampuan kartu pada data uji**")
        metrik = pd.DataFrame({
            "Ukuran": list(INFO["metrik_uji"].keys()),
            "Nilai": [f"{v:.4f}" for v in INFO["metrik_uji"].values()],
        })
        st.dataframe(metrik, width="stretch", hide_index=True)
        st.caption(
            f"Dihitung pada {INFO['data']['baris_uji']:,} mahasiswa yang tidak pernah "
            f"dipakai membangun kartu. Validasi silang lima lipatan menghasilkan ROC-AUC "
            f"{INFO['validasi_silang']['rata_rata']:.4f} dengan simpangan baku "
            f"{INFO['validasi_silang']['simpangan_baku']:.4f}."
        )

    with kolom_b:
        st.markdown("**Harga transparansi**")
        st.markdown(
            f"""
            Model Gradient Boosting yang dilatih pada seluruh
            {len(ksk.SELURUH_KOLOM)} kolom tanpa satu pun pembatasan mencapai ROC-AUC
            {INFO['metrik_pembanding']['gradient_boosting_roc_auc']:.4f}, sedangkan kartu
            skor ini {INFO['metrik_uji']['ROC-AUC']:.4f}. Selisihnya
            {INFO['metrik_pembanding']['harga_transparansi_roc_auc']:.4f} poin.

            Selisih sekecil itu ditukar dengan tiga hal: setiap poin dapat ditelusuri,
            kartu dapat dipakai tanpa perangkat apa pun, dan setiap barisnya dapat
            diperdebatkan oleh bagian akademik.
            """
        )
        st.markdown("**Profil mahasiswa berisiko**")
        for nomor, isi in INFO["profil"].items():
            st.markdown(f"- **{isi['nama']}**: {isi['keterangan']}")

    st.markdown("**Batasan yang perlu diketahui pemakai**")
    st.markdown(
        """
        - Kartu memberi urutan perhatian, bukan vonis. Skor rendah berarti mahasiswa
          pantas ditanyai kabarnya lebih dulu, bukan bahwa ia pasti berhenti.
        - Hubungan yang dipelajari bersifat keterkaitan, bukan sebab akibat. Tunggakan
          uang kuliah bisa jadi gejala dari keputusan berhenti yang sudah diambil, bukan
          penyebabnya.
        - Jenis kelamin sengaja dikeluarkan dari kartu meskipun secara angka berguna,
          karena tidak patut dipakai untuk memberi perlakuan berbeda kepada perorangan.
        - Profil mahasiswa berisiko adalah alat bantu percakapan, bukan label permanen.
          Sebagian mahasiswa berada di perbatasan antar profil.
        - Kartu perlu disusun ulang setiap tahun akademik, karena kurikulum, kebijakan
          biaya, dan profil pendaftar berubah dari waktu ke waktu.
        """
    )

    st.markdown(
        f"""
        <div class="catatan">
        Sumber data: <a href="{INFO['data']['sumber']}">{INFO['data']['sumber']}</a>.
        Kartu versi {INFO['identitas']['versi']}, disusun ulang lewat notebook.ipynb pada
        proyek yang sama.
        </div>
        """,
        unsafe_allow_html=True,
    )
