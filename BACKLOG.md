# RoamWise Backlog

> **2026-08-22 güncellemesi:** Bu dosya artık GitHub Issues ile **senkronize edilmiş bir
> özet/indeks**. Kaynak-doğru (source of truth) artık doğrudan
> [GitHub Issues](https://github.com/yukselburcinn-web/DA592/issues) — yeni bir işi buraya
> yazıp sonra Issue açmak yerine, doğrudan GitHub'da Issue açın; bu dosyayı daha sonra
> senkronize ederiz. İlk sürümdeki 12 maddenin tamamı zaten Issue'ya dönüştürülüp
> çoğu kapatılmış durumda; #13-32 arası (bazı numaralar PR'lara ait) bu ilk listeden sonra
> doğrudan GitHub'da açılmış yeni işler.

Alan sahipliği önerisi (isteğe göre değiştirin):
- **Alan A — Veri & Modeller**: `data/`, `models/`
- **Alan B — Retrieval & Bilgi Grafiği**: `retrieval/`, `knowledge_graph/`
- **Alan C — Ajanlar, Orkestrasyon & UI**: `agents/`, `app.py`, `evaluation/`
- **Alan D — Altyapı**: CI/CD, deployment, routing/transit altyapısı, import/paket düzeni

Gruplama işin **içeriğine** göre yapıldı; bazı GitHub etiketleri (`area:*`) içerikle tam
örtüşmüyor (ör. #27/#30/#31 hepsi `area:agents` etiketli ama sırasıyla veri, veri ve
altyapı işi) — bu durumlarda not düşüldü, etiketler GitHub'da düzeltilebilir.

## Nasıl kullanılır

1. Yeni bir iş fikri için doğrudan GitHub'da **New Issue** açın (aşağıdaki maddelerin
   formatını örnek alın: kısa özet + kabul kriterleri).
2. Etiketleri GitHub'da yoksa önce oluşturun (Settings → Labels): `area:data`, `area:retrieval`,
   `area:agents`, `area:ui`, `area:infra`, `priority:high`, `priority:medium`, `priority:low`,
   `good first issue`, `bug`, `enhancement`.
3. Bu dosyayı periyodik olarak `gh issue list --state all` çıktısıyla senkronize edin.

---

## Durum özeti (2026-08-22)

| Alan | Açık | Kapalı |
|---|---|---|
| A — Veri & Modeller | 1 (#30) | 4 (#1, #2, #3, #27) |
| B — Retrieval & Bilgi Grafiği | 0 | 3 (#4, #5, #6) |
| C — Ajanlar, Orkestrasyon & UI | 6 (#7, #19→kapalı, #20→kapalı, #21, #22, #23, #29) | 4 (#8, #9, #10, #19, #20) |
| D — Altyapı | 3 (#26, #31, #32) | 2 (#11, #12) |

İlk 12 maddelik listenin **10'u kapalı** — proje büyük ölçüde ilk backlog'u bitirmiş durumda.
Açık kalan işlerin çoğu (#21-23, #29-31) ilk listede yoktu, uygulama canlı test edilirken/
kod incelenirken ortaya çıkan yeni bulgular.

---

## Alan A — Veri & Modeller

### #1. Sentetik veriyi gerçek kaynaklarla değiştir — ✅ Kapalı
`data/generate_data.py`'nin sentetik POI/talep verisini gerçek kaynaklarla değiştirme işiydi.
[#1](https://github.com/yukselburcinn-web/DA592/issues/1) → sonradan #27 ile daha da derinleştirildi.

### #2. POI kataloğunu şehir başına büyüt — ✅ Kapalı
10 POI/şehirden başlayıp kademeli büyütüldü (bugün 1200 POI, 150/şehir).
[#2](https://github.com/yukselburcinn-web/DA592/issues/2)

### #3. Forecasting modelini genişlet / karşılaştır — ✅ Kapalı
Holt-Winters vs. Prophet backtest'i eklendi (`models/forecasting_prophet.py`,
`evaluation/forecasting_comparison.py`). [#3](https://github.com/yukselburcinn-web/DA592/issues/3)

### #27. POI verisini OpenStreetMap/Wikidata/Wikipedia'dan kaynaklandır — ✅ Kapalı
*(Etiket: `area:agents`, içerik: veri kaynağı — etiket içerikle örtüşmüyor.)*
Önceki "gerçek veri" iddiası aslında elle üretilmiş koordinat/puan/açıklama içeriyordu
(`random.uniform` saçılma, elle yazılmış 4.2-4.9 popülerlik puanı). Nominatim+Overpass+
Wikidata+Wikipedia'nın dördü de API key istemeden gerçek koordinat, kategori, açılış saati,
Wikidata QID ve Wikipedia özeti sağlayacak şekilde yeniden yazıldı; her POI artık bağımsız
doğrulanabilir bir Wikidata QID taşıyor, köken sütunları (`*_source`) hangi alanın gözlem
hangisinin varsayım olduğunu kaydediyor. Yan bulgu: OSM'nin `wikidata` etiketi bazen yere değil
kavrama işaret ediyor (Prag Hayvanat Bahçesi'ndeki panda çiti → tür maddesi) — koordinat
kapısıyla 1.685 böyle kayıt elendi. [#27](https://github.com/yukselburcinn-web/DA592/issues/27)

### #30. POI kataloğunda hâlâ eksik olan 5 ikonik yer — 🔓 Açık
*(Etiket: `area:agents`, içerik: veri/kategori kotası — etiket içerikle örtüşmüyor.)*
#28'in (main'e merge edilmiş şöhret-bazlı sıralama) ardından ikon kapsaması 20/47'den
42/47'ye çıktı ama Kapalıçarşı, Sacré-Cœur, Foro Romano, Alfama, Praça do Comércio hâlâ
listede yok — kategori kotasının round-robin katılığı (bir kategoride şöhret farkı büyükse
bile eşit sayıda alması) ve bazı OSM etiketlerinin (`place=square`, `place=neighbourhood`)
sorguya hiç girmemesi sebep gösteriliyor.
**Kabul kriterleri:**
- [ ] Kota, şöhret farkı büyük olduğunda esneyebiliyor, hiçbir kategori tamamen boşalmıyor
- [ ] `place=square` ve benzeri eksik etiketler değerlendiriliyor
- [ ] 47 ikonluk listede kapsama ≥45/47, şehir başına POI sayısı 150'de kalıyor

[#30](https://github.com/yukselburcinn-web/DA592/issues/30) — ilişkili: #28 (merge edilmiş)

---

## Alan B — Retrieval & Bilgi Grafiği

### #4. Semantic search'ü gerçek embedding modeline geçir — ✅ Kapalı
TF-IDF+LSA → `sentence-transformers` (`all-MiniLM-L6-v2`) + FAISS.
[#4](https://github.com/yukselburcinn-web/DA592/issues/4)

### #5. Neo4j'ye opsiyonel geçiş — ✅ Kapalı
`GraphIndex` artık `backend="neo4j"` parametresiyle opsiyonel bir Neo4j backend'i destekliyor
(varsayılan hâlâ NetworkX; canlı akışta hiçbir yerde `backend="neo4j"` ile çağrılmıyor).
[#5](https://github.com/yukselburcinn-web/DA592/issues/5)

### #6. Multi-hop sorgu seti genişlet — ✅ Kapalı
`evaluation/comparative_analysis.py::TEST_QUERIES` genişletildi.
[#6](https://github.com/yukselburcinn-web/DA592/issues/6)

---

## Alan C — Ajanlar, Orkestrasyon & UI

### #7. Gerçek LLM entegrasyonunu uçtan uca test et — 🔓 Açık
`AnthropicLLMClient` kodda var ama canlı API anahtarıyla hiç test edilmedi; hallucination
probe sonucu REPORT.md'ye eklenmedi. [#7](https://github.com/yukselburcinn-web/DA592/issues/7)

### #8. Rota optimizasyonunu gerçek yol ağıyla geliştir — ✅ Kapalı
Haversine → OSRM (`optimization/osrm_client.py`, opt-in `use_real_routing`), açılış saatleri
de itinerary'ye eklendi. [#8](https://github.com/yukselburcinn-web/DA592/issues/8) —
takibi: **#32** (OSRM demo sunucusunu self-hosted motora taşıma + transit).

### #9. LangGraph'a opsiyonel geçiş — ✅ Kapalı
`agents/orchestrator_langgraph.py`, aynı `plan_trip()` arayüzüyle.
[#9](https://github.com/yukselburcinn-web/DA592/issues/9)

### #10. Streamlit UI iyileştirmeleri — ✅ Kapalı
[#10](https://github.com/yukselburcinn-web/DA592/issues/10) — takibi: #21, #22, #23 (yeni UI bulguları).

### #19. Günlük rota dağılımını ve optimizasyonunu iyileştir — ✅ Kapalı
Gün başına dengesiz süre dağılımı + seyahat modu (yürüme/araç/hibrit) eksikliği giderildi
(`_fill_days_to_budget`, `_rebalance_days`, `optimization/travel_modes.py`).
[#19](https://github.com/yukselburcinn-web/DA592/issues/19)

### #20. Günlük itinerary'e minimum yemek yeri garantisi ekle — ✅ Kapalı
Her güne en az 2 "food" durağı garantisi (`_ensure_daily_meals`).
[#20](https://github.com/yukselburcinn-web/DA592/issues/20) — takibi: **#29** (zamanlama kümelenmesi).

### #21. Harita görünümünü ve ölçeklemesini güncelle — 🔓 Açık
`app.py::_fit_zoom` ve plotly harita bileşenleri; zoom itinerary alanına göre düzgün
ayarlanmıyor. [#21](https://github.com/yukselburcinn-web/DA592/issues/21)

### #22. Kullanıcıya gösterilmemesi gereken UI/debug bilgilerini temizle — 🔓 Açık
[#22](https://github.com/yukselburcinn-web/DA592/issues/22)

### #23. Budget ve Max price level slider'larını kaldır — 🔓 Açık
`good first issue`. [#23](https://github.com/yukselburcinn-web/DA592/issues/23)

### #29. Öğün durakları zamanda kümeleniyor — 🔓 Açık
#20/#25 ile öğün *sayısı* garantiye bağlandı ama 256 ölçülen günün 153'ünde (%60) iki öğün
2 saatten yakın (medyan ara 1.53 saat) — cheapest-insertion yerleşimi `MEAL_WINDOW_HOURS`
kadar kayınca iki öğün aynı boşluğa düşebiliyor. Ayrıca 1 günde (PRG/Nature/3 gün) tek öğünle
kalma kaçağı var.
**Kabul kriterleri:**
- [ ] İki öğün arası, gün penceresi izin verdiğince minimum eşiğin (öneri 3 saat) altına düşmüyor
- [ ] PRG/Nature/3-gün dahil, ölçülen 256 günün tamamında 2 öğün sağlanıyor
- [ ] Coğrafi maliyet #25 seviyesinde kalıyor (öğünler gidiş-dönüş sapması yaratmıyor)

[#29](https://github.com/yukselburcinn-web/DA592/issues/29)

---

## Alan D — Altyapı

### #11. CI pipeline kur — ✅ Kapalı
`.github/workflows/tests.yml`, her PR'da `pytest`. [#11](https://github.com/yukselburcinn-web/DA592/issues/11)

### #12. Dockerfile / deploy hazırlığı — ✅ Kapalı
[#12](https://github.com/yukselburcinn-web/DA592/issues/12)

### #26. Aynı modüller iki farklı import yolundan iki kez yükleniyor — 🔓 Açık
*(bug)* #6'daki `roamwise.*` mutlak import geçişi yarım kalmış: 7 dosya mutlak, 8 dosya göreli
import kullanıyor. Her iki yol da `sys.path`'te olduğundan aynı dosya **iki ayrı modül
nesnesi** olarak yükleniyor — `isinstance()` çapraz yollarda sessizce `False` dönüyor,
`monkeypatch`/cache/singleton state ikiye bölünüyor. Bugün gözle görülür bir hataya yol
açmıyor (henüz buna bağlı bir dal yok) ama teşhisi zor bir tuzak; **#31 ile aynı kök sebep**
(karışık import stili), farklı belirti.
**Kabul kriterleri:**
- [ ] Tüm iç importlar `roamwise.*` mutlak biçime geçirildi
- [ ] `sys.modules`'te hiçbir modül iki kez yüklü değil
- [ ] `streamlit run app.py` ve `pytest` PYTHONPATH hilesi olmadan çalışıyor

[#26](https://github.com/yukselburcinn-web/DA592/issues/26)

### #31. Uygulama dokümante edilen hiçbir komutla açılmıyor (`ModuleNotFoundError`) — 🔓 Açık
*(Etiket: `area:agents`, içerik: import/altyapı sorunu — etiket içerikle örtüşmüyor.
**#26 ile aynı kök sebep**, farklı belirti: bu issue somut hatayı ve üç dokümante edilmiş
çalıştırma yolunun üçünün de kırık olduğunu gösteriyor.)*
`README.md`, `.claude/launch.json` ve `Dockerfile`'daki komutların üçü de Streamlit'in
`sys.path[0]`'a script dizinini (`roamwise/`) eklemesi, repo kökünü eklememesi yüzünden
`ModuleNotFoundError: No module named 'roamwise'` ile açılmıyor — bunu bu oturumda da
birebir yaşadık, `PYTHONPATH=$(pwd)` ile geçici çözdük. Testler yeşil kalıyor çünkü pytest
`roamwise/__init__.py` sayesinde repo kökünü otomatik `sys.path`'e ekliyor; Streamlit bunu
yapmıyor — yani **test paketi bu hata sınıfını yapısal olarak göremiyor**.
**Kabul kriterleri:**
- [ ] Kod tabanı tek bir import stiline getirildi (bkz. #26)
- [ ] README/launch.json/Dockerfile komutları ek ortam değişkeni gerektirmeden çalışıyor
- [ ] Uygulamanın import edilebildiğini doğrulayan bir smoke test var

[#31](https://github.com/yukselburcinn-web/DA592/issues/31)

### #32. OSRM demo sunucusunu self-hosted motora taşı + transit desteği — 🔓 Açık
`osrm_client.py` genel-erişimli, garantisiz bir demo sunucuya bağlanıyor; sistemde hiç toplu
taşıma modellemesi yok. Ayrıca knowledge graph'taki 16 gerçek transport hub'ı (`data/transport.csv`)
rota optimizasyonunda hiç kullanılmıyor — `RouterAgent` ilk günü şehir merkezinden başlatıyor,
havalimanı/istasyon koordinatından değil. Önerilen çözüm: (1) Valhalla/GraphHopper ile
self-hosted routing (düşük efor), (2) OpenTripPlanner + GTFS ile 1-2 pilot şehirde transit
(yüksek efor), (3) transport hub'ı 1. gün rotasına gerçekten bağlama. Google Maps API bilinçli
olarak değerlendirilmedi: Places/Directions içeriğinin kalıcı önbelleklenmesi ToS ile yasak,
projenin "bir kere çek, statik kullan" mimarisiyle uyuşmuyor.
**Kabul kriterleri:** issue gövdesinde detaylı — bkz. link.

[#32](https://github.com/yukselburcinn-web/DA592/issues/32)

---

## Öncelik sırası (2026-08-22 itibarıyla açık işler)

| Öncelik | Issue | Neden |
|---|---|---|
| 1 | #31 + #26 | Aynı kök sebep, uygulama şu an dokümante edilen hiçbir komutla açılmıyor — yeni katılan biri ilk denemede takılır |
| 2 | #7 | `priority:high` etiketli, final rapor için gerçek LLM sonucu eksik |
| 3 | #29 | Ölçülmüş, somut bir kalite sorunu (%60 gün etkileniyor) |
| 4 | #21, #22, #23 | UI cilası, düşük risk/düşük efor |
| 5 | #30 | Veri kalitesi ince ayarı |
| 6 | #32 | Kapsamlı altyapı işi — Aşama 1 (Valhalla/GraphHopper) tek başına da değerli, Aşama 2 (transit) stretch goal |
