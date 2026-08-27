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

## Durum özeti (2026-08-26, akşam)

| Alan | Açık | Kapalı |
|---|---|---|
| A — Veri & Modeller | 4 (#30, #33, #71, #79) | 6 (#1, #2, #3, #27, #65, #70) |
| B — Retrieval & Bilgi Grafiği | 1 (#49) | 8 (#4, #5, #6, #42, #46, #48, #50, #63) |
| C — Ajanlar, Orkestrasyon & UI | 6 (#7, #76, #77, #78, #80, #81) | 17 (#8, #9, #10, #19, #20, #21, #22, #23, #29, #41, #54, #56, #57, #59, #61, #67, #72) |
| D — Altyapı | 1 (#32) | 4 (#11, #12, #26, #31) |

İlk 12 maddelik listenin tamamı kapalı — proje ilk backlog'u bitirdi. Açık kalan işlerin hepsi ilk
listede yoktu; uygulama canlı test edilirken ya da kod incelenirken çıkan bulgular.

Toplam 47 issue: **12 açık, 35 kapalı.** Sayılar ve durumlar 2026-08-26 itibarıyla `gh issue list`
ile doğrulandı.

**Bu güncellemede ne değişti:** #70 ve #72 kapandı. #72 (TOPTW router) altı yeni bulgu doğurdu —
#76–#81 — ve iki issue'nun (#32, #71) varsayımlarını geçersiz kıldığı için ikisi de yeniden yazıldı.
Açık iş sayısı 8'den 12'ye çıktı; bu bir gerileme değil, tek bir büyük issue'nun içinden çıkan
işlerin ayrıştırılması.

---

## Alan A — Veri & Modeller

### #1. Sentetik veriyi gerçek kaynaklarla değiştir — ✅ Kapalı
`data/generate_data.py`'nin sentetik POI/talep verisini gerçek kaynaklarla değiştirme işiydi.
[#1](https://github.com/yukselburcinn-web/DA592/issues/1) → sonradan #27 ile daha da derinleştirildi.

### #2. POI kataloğunu şehir başına büyüt — ✅ Kapalı
10 POI/şehirden başlayıp kademeli büyütüldü; katalog daha sonra sekiz şehirde 150 POI/şehirden iki şehirde 400 (Paris) + 300 (Berlin) POI'ye taşındı — toplam 700. Şehir başına derinlik, Wikivoyage referans listesine karşı kapsamayı Paris'te %38'den %75'e çıkardı.
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

### #65. Katalogdaki POI'lerin %10'u gezilebilir bir yer değil — ✅ Kapalı
*(bug, `priority:high`)* 700 POI'nin **71'i (%10.1)** bir gezginin uğrayabileceği yer değildi:
üniversiteler, bir hastane, metro istasyonları, yıkılmış binalar, bir televizyon kanalı, bir yangın.
Teorik değil — gerçekten plana giriyorlardı (Sorbonne → Sciences Po → … → Bataclan bir "kültür günü"
olarak dönüyordu). `pipeline/sight_filter.py` ile ayıklandı, elenenler `data/dropped_pois.csv`'ye
kaydedildi, katalog 700 → 654'e indi ve REPORT'taki sayılar buna göre güncellendi.
[#65](https://github.com/yukselburcinn-web/DA592/issues/65)

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

### #70. Açılış saatleri parse'ı OSM etiketinin %92'sinde bilgi kaybediyor — ✅ Kapalı
*(bug, `priority:high`)* `build_catalogue.py::parse_opening_hours` bir regex; `opening_hours`
etiketindeki ilk `HH:MM-HH:MM`'i alıp gerisini atıyor. Deponun kendi Overpass önbelleğindeki
**4.404 eşsiz etiket** üzerinden ölçüldü: %94'ü gün bilgisi içeriyor, %53'ü çok kurallı, %17'sinde
öğle arası var, %11'inde açık `off` kuralı — ve **%92'sinde bilgi kayboluyor**, %5'i hiç parse
edilemiyor.

Sonucu: **haftanın günü hiç taşınmıyor**, yani pazartesi kapalı bir müzeyi pazartesiye koymak bugün
mümkün. Parse edilemeyenler daha kötü — `closed` ya da
`"Fermé pour travaux jusqu'à novembre 2028"` etiketli POI sessizce kategori varsayılanı saatlerini
alıyor, yani kapalı bir yer 09:00–18:00 açık görünüyor.

Rota kalitesi için en yüksek getirili veri işi ve **yeni kaynak gerektirmiyor**: etiket zaten
çekiliyor, parse'ta atılıyor. OSM ODbL — commit edilebilir. Şema additive tutulmalı; `poi.csv`'nin
19 kolonu bozulursa bilgi grafiği ve retrieval sessizce çöker.

**Sonuç:** `opening-hours-py` ile değiştirildi, katalogdaki 4.404 eşsiz etiketin %99.4'ü parse
ediliyor. Haftanın günü artık taşınıyor: haftanın her gününe başlayan geziler üzerinde ölçülen
455 durağın 38'i kapalı bir güne düşüyordu, şimdi **sıfır**. #72 bunun üzerine kuruldu — TOPTW
tamamen zaman pencerelerine dayanıyor. Kalan boşluk kapsam: POI'lerin yalnızca %39.8'inde gerçek
OSM etiketi var, gerisi kategori varsayımı — bu artık #71'in konusu.

[#70](https://github.com/yukselburcinn-web/DA592/issues/70) · [PR #74](https://github.com/yukselburcinn-web/DA592/pull/74)

### #71. Açılış saati ve fiyat kapsamını kapat: Google Maps scraping dahil kaynak kararı — 🔓 Açık
*(enhancement, `priority:high` — 2026-08-26'da `low`'dan yükseltildi ve yeniden yazıldı)* Karar
issue'su, uygulama değil. Kapsam Places API'sinden **Google Maps scraping**'i de içerecek şekilde
genişletildi.

#70 ve #72 kapandığı için karar artık ölçülebilir. Kataloğun 654 POI'sinde açılış saati **260'ında
(%39.8)** gerçek OSM etiketi, gerisi kategori varsayımı; fiyat **80'inde (%12.2)** gerçek. İki somut
sonuç: router'ın kısıtlarının %60'ı tahmin (TOPTW tamamen zaman pencerelerine dayanıyor), ve bütçe
slider'ı hiçbir şey ifade edemiyor (`price_level` tek bit, 61 food POI'nin hepsi aynı değerde).

**Scraping engeli kaldırmıyor, büyütüyor:** asıl engel API mekanizması değil, `poi.csv`'nin public
depoda commit'li olması. Arayüz kazıma ToS'ta ayrıca yasak, sayfa yapısına bağlı olduğu için
kırılgan, ve REPORT'un her alanın kaynağını beyan eden yapısıyla çelişiyor. İzin veren alternatifler
(Overture CDLA-Permissive, Foursquare Apache-2.0) aynı iki boşluğu kapatabilir — **kapsamlarını
ölçmek yarım günlük iş ve sıfır risk**, o ölçüm yapılmadan scraping'e geçmek ihlali gereksiz yere
üstlenmek olabilir. [#71](https://github.com/yukselburcinn-web/DA592/issues/71)

### #33. Talep tahminine ve fiyat sinyaline şehir düzeyinde granülerlik ekle — 🔓 Açık
*(enhancement, `priority:medium`)* Talep verisi Eurostat `tour_occ_nim`'den geliyor: gerçek, aylık,
COVID çöküşünü içeren bir seri — ama granülerliği **ülke düzeyinde**. Yani Paris'in talep tahmini
tüm Fransa'nın turist sayısını proxy alıyor. REPORT §5'in kendi ifadesiyle "tek dürüst kalan boşluk".
Eurostat'ın şehir serisi (`urb_ctour`) var ama **yıllık**, aylık mevsimsellik varsayan Holt-Winters'ı
besleyemiyor.

İkinci yarısı fiyat: `price_level` ile crowding tahmini hiçbir yerde birbirine bağlanmıyor, yani
proposal'ın "bütçeyi mevsimsel talebe göre hizala" çerçevesinin sadece crowding tarafı gerçek veriyle
karşılanıyor. Önerilen kaynak: Inside Airbnb. [#33](https://github.com/yukselburcinn-web/DA592/issues/33)

### #79. Macera slider'ı puanı hiç hareket ettirmiyor — 🔓 Açık
*(enhancement, `priority:medium`)* #72'nin puan fonksiyonu, tercih vektörünü kategori ağırlıklarına
çeviren matrisi `user_survey.csv` ile `CATEGORY_AFFINITY` arasından NNLS ile türetiyor. Türetilen
matriste **`adventure` satırının tamamı sıfır**: kullanıcı slider'ı nereye çekerse çeksin puan
değişmiyor.

Sebep taksonomi: katalogda maceraya karşılık gelen kategori yok, en yakın aday `beach` ve iki
şehirlik sette `beach` kategorili **sıfır POI** var. Ankette `adventure` ile `nature` korele olduğu
için NNLS ortak varyansı tamamen `nature`'a veriyor.

Aynı sınıftan ikinci sorun: bütçe slider'ı da fiyat verisi tek bit olduğu için etkisiz (#71). Yani
**altı slider'ın ikisi bugün çalışmıyor.** Taksonomi 10 kategoride sabit ve retrieval buna bağımlı —
kategori eklenirse `CATEGORY_PHRASE`, `CATEGORY_AFFINITY` ve graf birlikte güncellenmeli.
[#79](https://github.com/yukselburcinn-web/DA592/issues/79)

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

### #42. Retrieval config seçicisini ön yüzden kaldır, karşılaştırma sekmesi ekle — ✅ Kapalı
Sidebar'daki fusion/hybrid/standard radio'su bir deney knob'uydu, gezginin anlamlı seçim
yapabileceği bir şey değil. Fusion sabitlendi; karşılaştırma **System logs → Results** sekmesine
taşındı, orada soru değil kanıt olarak duruyor. [#42](https://github.com/yukselburcinn-web/DA592/issues/42)

### #46. Karşılaştırmalı analize istatistiksel anlamlılık testi ekle — ✅ Kapalı
Ortalamalar tek başına hangi farkın gerçek, hangisinin gürültü olduğunu söylemiyordu. Sonuçlar eşli
(her config aynı sorguları görüyor), dolayısıyla Wilcoxon signed-rank uygulandı — `scipy` zaten
bağımlılıktı. Sonuç: Fusion, Hybrid'e karşı **arketip kesinliğinde** anlamlı, multi-hop recall ve
km/durak'ta **ayırt edilemez**. [#46](https://github.com/yukselburcinn-web/DA592/issues/46)

### #48. recall_at_k kısmen döngüseldi: gold set Fusion'ın kendi retriever'ından üretiliyordu — ✅ Kapalı
Gold set, Fusion'ın graph retriever'ıyla aynı fonksiyondan geliyordu; metrik kısmen kendi kendini
not veriyordu. Değerlendirme bağımsız bir kaynağa (Wikivoyage) taşındı.
[#48](https://github.com/yukselburcinn-web/DA592/issues/48)

### #50. Test sorgu setini genişlet ve bağımlılık seviyesini raporla — ✅ Kapalı
Set çok dar ve dengesizdi (56 hücrenin 14'ü dolu). 55 sorguya çıkarıldı: 19'u elle yazılmış, 36'sı
şehir × kategori hücrelerini düzgün tarayan üretilmiş sorgu. Genişletmenin bedeli dürüstçe
raporlandı — **eski setin gösterdiği multi-hop recall üstünlüğü kayboldu**, arketip kesinliği ise
iki katmanda da tuttu. [#50](https://github.com/yukselburcinn-web/DA592/issues/50)

### #63. Öneriler alakasız: Louvre yerine 'France 3' (TV kanalı) — ✅ Kapalı
Kültür gezgini Paris sorgusunda 24 adayın 7. sırasında bir televizyon kanalı vardı; Louvre 17.,
Eyfel/Notre-Dame/Sacré-Cœur/Père Lachaise listede hiç yoktu. Sıralama gezginin görmek isteyeceğine
göre yeniden kuruldu ve ünlülük eş-değer bir faktör değil **çarpan** hâline getirildi.
[#63](https://github.com/yukselburcinn-web/DA592/issues/63)

### #49. recall_at_k yapısal olarak tavanlı, ama 1.0 üzerinden okunuyor — 🔓 Açık
*(enhancement, `priority:medium`)* #46'daki değerlendirmenin ikinci bulgusu (ilki #48). `recall_at_k`
k=8'de ölçülüyor ama medyan sorgunun gold set'i çok daha büyük, dolayısıyla **kusursuz sıralayan bir
retriever bile 1.0'a ulaşamaz** — ulaşılabilir tavan ~0.573. Sayı bugün 1.0 üzerinden okunuyor ve
Fusion'ın skoru olduğundan kötü görünüyor. Metrik ya tavanına göre normalize edilmeli ya da tavan
raporda açıkça yazılmalı. [#49](https://github.com/yukselburcinn-web/DA592/issues/49)

---

## Alan C — Ajanlar, Orkestrasyon & UI

### #7. Gerçek LLM entegrasyonunu uçtan uca test et — 🔓 Açık
`AnthropicLLMClient` kodda var ama canlı API anahtarıyla hiç test edilmedi; hallucination
probe sonucu REPORT.md'ye eklenmedi. [#7](https://github.com/yukselburcinn-web/DA592/issues/7)

### #8. Rota optimizasyonunu gerçek yol ağıyla geliştir — ✅ Kapalı
Haversine → gerçek yol ağı (opt-in `use_real_routing`), açılış saatleri de itinerary'ye
eklendi. [#8](https://github.com/yukselburcinn-web/DA592/issues/8) — takibi: **#32**; oradaki
Aşama 1 ile `osrm_client.py` kalktı, yerine `optimization/street_network.py` geçti.

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

### #21. Harita görünümünü ve ölçeklemesini güncelle — ✅ Kapalı
`app.py::_fit_zoom` ve plotly harita bileşenleri; zoom itinerary alanına göre düzgün
ayarlanmıyor. [#21](https://github.com/yukselburcinn-web/DA592/issues/21)

### #22. Kullanıcıya gösterilmemesi gereken UI/debug bilgilerini temizle — ✅ Kapalı
[#22](https://github.com/yukselburcinn-web/DA592/issues/22)

### #23. Tercihleri Düşük/Orta/Yüksek seçimine çevir + Max price level filtresini kaldır — ✅ Kapalı
`good first issue`. [#23](https://github.com/yukselburcinn-web/DA592/issues/23)

### #29. Öğün durakları zamanda kümeleniyor — ✅ Kapalı
#20/#25 ile öğün *sayısı* garantiye bağlandı ama 256 ölçülen günün 153'ünde (%60) iki öğün
2 saatten yakın (medyan ara 1.53 saat) — cheapest-insertion yerleşimi `MEAL_WINDOW_HOURS`
kadar kayınca iki öğün aynı boşluğa düşebiliyor. Ayrıca 1 günde (PRG/Nature/3 gün) tek öğünle
kalma kaçağı var.
**Kabul kriterleri:**
- [ ] İki öğün arası, gün penceresi izin verdiğince minimum eşiğin (öneri 3 saat) altına düşmüyor
- [ ] PRG/Nature/3-gün dahil, ölçülen 256 günün tamamında 2 öğün sağlanıyor
- [ ] Coğrafi maliyet #25 seviyesinde kalıyor (öğünler gidiş-dönüş sapması yaratmıyor)

[#29](https://github.com/yukselburcinn-web/DA592/issues/29)

### #41. "Agent trace" sekmesini kaldır, ayrı bir log ekranı ekle — ✅ Kapalı
Orkestrasyonun ham `st.json` çıktısı gezginin itinerary'sinin ortasında duruyordu. Yerine
operatöre dönük **System logs** ekranı geldi: her ajan adımı süresiyle, seviye/metin filtreli ve
`.log` olarak indirilebilir. [#41](https://github.com/yukselburcinn-web/DA592/issues/41)

### #54. API anahtarı gerektirmeyen yerel LLM yolu ekle — ✅ Kapalı
Gerçek jeneratif çıktının tek yolu ücretli bir API anahtarıydı (#7'nin takıldığı yer). MLX üzerinden
yerel/açık ağırlıklı bir yol eklendi (`ROAMWISE_LOCAL_LLM=1`), böylece maliyet ödemeden gerçek —
template olmayan — LLM çıktısı görülebiliyor. [#54](https://github.com/yukselburcinn-web/DA592/issues/54)

### #56. Agent narrative rotada olmayan mekanları öneriyor — ✅ Kapalı
Gerçek LLM ile çalıştırıldığında anlatı, router'ın ürettiği rotayla uyuşmayan yerlerden bahsediyordu.
Model zayıflığı değil **prompt tasarımı hatası**ydı; `TemplateLLMClient` promptu aynen yankıladığı
için bugüne kadar görünmemişti. Anlatı artık yalnızca itinerary'den üretiliyor.
[#56](https://github.com/yukselburcinn-web/DA592/issues/56)

### #57. Gerçek LLM ile yanıt süresi çok uzun — ✅ Kapalı
Tek bir "Plan my trip" isteği 2 günlük gezide ~99 sn, 3 günlükte 400 sn'yi aşıp timeout'a düşüyordu:
istek başına 4 generation üretiliyor, **2'si hiç gösterilmiyordu**. Okunmayan generation'lar
kaldırıldı. #56 ile aynı PR'da çözüldü. [#57](https://github.com/yukselburcinn-web/DA592/issues/57)

### #59. Günün zaman modeli gerçekçi değil — ✅ Kapalı
Üç belirti, tek kök: gün sabit 09:00'da başlıyor, süre aralığı dar, ve kategorilerin günün hangi
saatine ait olduğu hiç modellenmemiş. Başlangıç saati UI'a çıkarıldı ve orchestrator üzerinden
geçirildi, süre 12–18 saate alındı, nightlife `NIGHTLIFE_EARLIEST_HOUR` ile günün sonuna taşındı.
Gece yarısı kırpması bilinçli olarak açık bırakıldı — takibi **#61**. [#59](https://github.com/yukselburcinn-web/DA592/issues/59)

### #67. Bütçe fiyat filtresi ölü koddu — ✅ Kapalı
`max_price_level=3` "1=budget, 3=splurge" diye belgelenmişti ama `price_level` OSM'in `fee` etiketi,
yani kademe değil ücretsiz/ücretli bayrağı ([0, 1]); eşik 3 olduğu için filtre **bugüne kadar tek bir
POI'yi bile elemedi**. Filtre silindi; katalogun fiyat hakkında bildiği dürüst kalıntı (ücretsiz
giriş oranı) filtrelenmek yerine raporlanıyor. [#67](https://github.com/yukselburcinn-web/DA592/issues/67)

### #61. Gece yarısını aşan kapanış saatleri ve günün başlangıcı — ✅ Kapalı
*(bug, `priority:high`, #59'un takibi)* İki ayrı kalıntı, tek semptom: nightlife ilgisi yüksek
seçilince günler tek bir mekânla dönüyordu.

1. **Gece yarısı kırpması.** `effective_close = 24.0 if close_h < open_h else close_h` — #59 bunu
   bilinçli olarak açık bırakıp REPORT.md'ye not düşmüştü. 02:00'ye kadar açık bir bar, 06:00'ya
   kadar süren bir günde bile 01:00'de "kapalı" sayılıyordu. Ölçüldü: 12:00 başlangıçlı 15 ve 18
   saatlik günler birebir aynı planı veriyordu, ikisi de 23:49'da duruyordu.
2. **Günün varsayılan başlangıcı.** #59 başlangıç saatini ayarlanabilir yaptı ama 7–12 ile
   sınırladı ve varsayılanı 09:00 bıraktı. Nightlife 18:00'den önce planlanmadığı için
   (#59'un doğru kuralı) 09:00'da açılan günün ilk dokuz saati yapısal olarak boş kalıyor.

**Çözüm:** kapanış saati günün kendi saatinde 24'ü aşabiliyor (`_opening_intervals`,
`_next_open_hour`); başlangıç saati arketipten geliyor (`router_agent.DAY_START_HOURS`,
Nightlife Seeker → 15:00), UI'da "otomatik" varsayılan + 7–18 arası elle seçim. Ayrıca gün
doluluğu geçen saatle değil gerçekten meşgul geçen süreyle ölçülüyor (`active_minutes` /
`idle_minutes`).

**Ölçüm (36 gün, 2 şehir × 2 profil × {12,15,18} saat; #60'ın 09:00 varsayılanına karşı):**

| | #60 | #61 |
|---|---|---|
| Tek duraklık gün (nightlife=High) | 5/18 | **0/18** |
| Durak/gün (High) | 2.33 | **4.56** |
| Programlanan nightlife durağı | 21 | **56** |
| 12 saatlik günler (High) | 1,1,1,1,2,1 | **5,5,5,4,4,3** |
| 24:00 sonrası durak | 0 | **8** (en geç 03:53) |
| Culture Enthusiast (regresyon kontrolü) | 5.44 | 5.44 |

Kapsam dışı bırakıldı: genel zaman-pencereli sıralama. #59'un kategori kuralı + bekle/atla geçişi
korundu; geç açılan bir müze hâlâ aynı muameleyi görmüyor. Gerçek çözüm OR-Tools CP-SAT tipi bir
TSPTW formülasyonu — REPORT §5'te açık bırakıldı.

[#61](https://github.com/yukselburcinn-web/DA592/issues/61)

### #72. Rota modelini TSP'den TOPTW'ye taşı — ✅ Kapalı
*(enhancement, `priority:medium`)* Router bir TSP çözüyordu: hepsini sırala, mesafeyi küçült,
sığmayanı ele. Turistin problemi ise **seçim** — hangi duraklar. TOPTW'ye geçildi ve altı yama
(`_fill_days_to_budget`, `_rebalance_days`, `_ensure_daily_meals`, `_ensure_evening_stops`,
`NIGHTLIFE_EARLIEST_HOUR`, `_nightlife_last`) tek bir modelin kısıtı hâline geldi.

**Issue'nun açık bıraktığı soru cevaplandı: takas gerçek değilmiş.** Greedy yerleştirme +%6 durak
karşılığında +%12 km/durak istiyordu ve "gerçek çözücü bunu ceza ödemeden alabilir mi" doğrulanmamıştı.
Alabiliyor — iki metrik aynı anda iyileşiyor. 72 gün, aynı adaylar, aynı coğrafya:

| havuz | | durak/gün | km/durak | iki öğünlü gün |
|---|---|---|---|---|
| 24 (eski varsayılan) | eski router | 6.21 | 1.582 | 28/72 |
| | TOPTW | **6.69** (+%7.8) | **0.906** (−%42.7) | **72/72** |
| 72 (**yeni varsayılan**) | eski router | 7.82 | 1.054 | 42/72 |
| | TOPTW | **9.06** (+%15.8) | **0.471** (−%55.3) | **72/72** |

Kişiselleştirme seçime taşındı: puan, gezginin altı slider'ını **doğrudan** okuyor, arketip
etiketine çökmeden — aynı "Culture Enthusiast" etiketine düşen iki gezgin artık farklı plan alıyor.
Puanı çözücü ağırlığı olarak da kullanmak denendi ve ölçülerek elendi: durak ve mesafe kaybettiriyor,
kendi amaç fonksiyonunu bile ancak %4 oynatıyor.

Kapatılmayan iki kriter ayrı issue'ya taşındı (#77, #78). Formülün `kalabalık_indirimi(tahmin, saat)`
parçası bugünkü veriyle karşılanamıyor — forecaster şehir-ay düzeyinde tek skaler döndürüyor, sabit
çarpan hiçbir seçimi değiştiremez; #33'e bağımlı.

[#72](https://github.com/yukselburcinn-web/DA592/issues/72) · [PR #75](https://github.com/yukselburcinn-web/DA592/pull/75)

---

### #76. LangGraph orchestrator router'a `start_date` geçirmiyor — 🔓 Açık
*(bug, `priority:medium`)* `orchestrator.py` geçiriyor, `orchestrator_langgraph.py` geçirmiyor;
`PlanState`'te alan bile yok. Sonucu: **#70'in kapattığı hata bu yolda açık** — tarih olmadan
`_opening_intervals` kaba `open_hour`/`close_hour` çiftine düşüyor, yani pazartesi kapalı müze
pazartesiye planlanabiliyor. #72'den sonra bedeli büyüdü: yanlış pencere artık yanlış gün ataması ve
yanlış saat demek. Mevcut eşdeğerlik testi iki orkestratörün **arayüzünü** karşılaştırıyor, çıktısını
değil — fark bu yüzden testten kaçtı. [#76](https://github.com/yukselburcinn-web/DA592/issues/76)

### #77. Toplanan puanı ulaşılabilir tavana karşı raporla — 🔓 Açık
*(enhancement, `priority:medium`)* #72'nin kapatılmamış kriteri. TOPTW bir puan maksimizasyonu
çözüyor ama çıktısında "ne kadar iyi" sorusunun cevabı yok; raporlanan durak sayısı ve km/durak
**tavansız**. 9 durak iyi mi? Bu havuzda 11 mümkünse kötü. #49'un retrieval tarafında tespit ettiği
hatanın birebir aynısı (`recall_at_k` ~0.573 tavanlı ama 1.0 üzerinden okunuyor) — ikisi birlikte
ele alınırsa tutarlı bir "neyin yüzdesi" anlatısı çıkar.
[#77](https://github.com/yukselburcinn-web/DA592/issues/77)

### #78. Kilitle-ve-yeniden-çöz: "bu durağı değiştir" — 🔓 Açık
*(enhancement, `priority:low`)* #72'nin kapatılmamış kriteri ve kısıt modelinin doğal kazancı:
bir durağı kilitlemek o POI'nin gün kopyasının `ActiveVar`'ını 1'e sabitlemek, reddetmek tüm
kopyalarını 0'a sabitlemek — ikisi de tek satır. Eksik olan model değil, arayüz ve akış. Dikkat
edilecek nokta: reddedildikçe havuz daralıyor ve öğün/doluluk kısıtları karşılanamaz hâle gelebilir;
sessizce gevşemek yerine bunu söylemek gerekir.
[#78](https://github.com/yukselburcinn-web/DA592/issues/78)

### #80. TOPTW formülasyonunun ~120 POI tavanı — 🔓 Açık
*(enhancement, `priority:low`)* Router adayları çözücüye vermeden önce puanla eliyor
(`MAX_WORKING_SET = 120`) ve bu bir tercih değil zorunluluk: 118 POI ~2 s'de çözülüyor, **371 POI
(tüm Paris kataloğu) 10 dakikada çözülmedi**. Sebep yapısal — her POI gün başına bir kopya alıyor
(propagation için gerekliydi), düğüm sayısı havuz × gün büyüyor.

Önemi: #72 puanın **aday seçici** olarak retrieval'ın arketip sorgusunu geçtiğini ölçtü; bunun
sonucu retrieval'ın ön filtrelemeyi bırakması olurdu, ama çözücü o ölçeğe çıkmıyor. Yollar: CP-SAT,
Vansteenwegen'in ILS'i, ya da kademeli eleme. Bugünkü kaliteyi engellemiyor.
[#80](https://github.com/yukselburcinn-web/DA592/issues/80)

### #81. POIZoner artık router yolunda değil — 🔓 Açık
*(`priority:low`)* #72 ile gün ataması modelin kararı oldu; `POIZoner` router tarafından hiç
çağrılmıyor. Modül ve üç testi duruyor, #19'un kapasite kısıtlı atama çalışması REPORT'ta anlatılıyor.
Hata değil, tercih sorusu: kalsın (proposal'ın "iki bağımsız KMeans" anlatısının parçası) mı,
kaldırılsın (çağrılmayan kod sonraki okuyucuyu yanıltır) mı. `TravelerSegmenter` etkilenmiyor.
[#81](https://github.com/yukselburcinn-web/DA592/issues/81)

---

## Alan D — Altyapı

### #11. CI pipeline kur — ✅ Kapalı
`.github/workflows/tests.yml`, her PR'da `pytest`. [#11](https://github.com/yukselburcinn-web/DA592/issues/11)

### #12. Dockerfile / deploy hazırlığı — ✅ Kapalı
[#12](https://github.com/yukselburcinn-web/DA592/issues/12)

### #26. Aynı modüller iki farklı import yolundan iki kez yükleniyor — ✅ Kapalı
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

### #31. Uygulama dokümante edilen hiçbir komutla açılmıyor (`ModuleNotFoundError`) — ✅ Kapalı
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

### #32. OSRM demo sunucusundan çık: Aşama 1 self-hosted mesafe, Aşama 2 GTFS transit — 🔓 Açık
*(enhancement, `priority:medium` — 2026-08-26'da yeniden yazıldı, 2026-08-27'de Aşama 1 kapandı)*
`osrm_client.py` garantisiz bir demo sunucuya bağlanıyordu; sistemde hâlâ hiç toplu taşıma
modellemesi yok.

**Eski gövde iki yanlış varsayım taşıyordu.** "8 şehir, 16 transport hub" — bugün **2 şehir, 18 hub**
(`transport.csv`: 3 havalimanı, 12 tren istasyonu, 3 otobüs terminali). Ve "OTP2 tek process = tek
graph, 8 şehir ya 8 container ya tek birleşik build" endişesinin sayısal temeli kalmadı: 2 şehir için
ikisi de makul. **Yani transit pilotu yazıldığından ucuz.** Router tarifi de eskimişti (POIZoner +
2-opt); entegrasyon noktası artık `routing._build_distance_functions` ve `fetch_distance_duration_matrix`
imzası korunduğu sürece router'a hiç dokunulmuyor. TOPTW matrisi gezi başına bir kez çekiyor, yani
rate-limit baskısı da düştü.

Üç bağımsız parça: **(1)** ✅ **transport hub 1. güne bağlandı** (aşağı bak);
**(2)** ✅ **Aşama 1 tamam** (aşağı bak);
**(3)** ✅ **Aşama 2 (Paris pilotu) tamam** (aşağı bak). Yolculuk planlama API'si
alternatifi bilinçli olarak elendi, tekrar gündeme getirilmesin.

**Aşama 1 (2026-08-27): B seçildi, A denenmedi.** `osrm_client.py` kalktı, yerine
`optimization/street_network.py` + `pipeline/build_street_network.py` geçti. Yol ağı OSM'den bir
kez indiriliyor, budanıyor ve `data/street_network/` altına commit'leniyor. **Karar gerekçesi:**
B (OSMnx + NetworkX) ölçülerek doğrulandı — Paris'te 868 yürüme çifti üzerinde OSRM ile ortalama
%4.3 (medyan %2.1) uyum; haversine aynı referanstan %16.4 sapıyor. A (self-hosted
Valhalla/GraphHopper) bu doğrulukla tek-container mimarisini `docker-compose` + kalıcı volume'a
çevirmeyi gerektirdiği için denenmedi.

`.graphml` yerine düz dizi `.npz`: Paris yürüme grafiği ham GraphML olarak 395 MB, aynı graf
budanmış dizi olarak 3.9 MB. Ayrıca **matris de önceden hesaplanıp commit'leniyor** — router'a
gelebilecek her nokta (POI + hub + gün başlangıcı olan şehir merkezi) zaten katalogda, yani çalışma
anında gerçek mesafe bir dizi okuması; grafik ise katalog dışı bir koordinat geldiğinde ikinci
katman olarak Dijkstra ile çözüyor. `use_real_routing` varsayılanı hâlâ kapalı, ama artık uptime
değil karşılaştırılabilirlik gerekçesiyle: REPORT'taki bütün ölçümler kapalıyken alındı.

**Parça 1 (2026-08-27): varış hub'ı 1. güne bağlandı.** `toptw.solve()` OR-Tools'a zaten araç
başına başlangıç düğümü veriyordu; ayrı bir varış düğümü eklendi ve yalnızca 0. aracın başlangıcı
ona bağlandı, diğer günler merkez deposundan devam ediyor. Zincir: kenar çubuğunda "Arriving at"
listesi → `plan_trip(arrival_hub_id=...)` → `RouterAgent.run()` (id'yi `city_transport`'tan
koordinata çeviriyor, tanınmayan id sessizce yok sayılıyor) → `build_multi_day_itinerary(arrival_hub=...)`.
Varsayılan **"Already in the city"**, yani mevcut davranış korunuyor: sessizce havalimanı seçmek
CDG'yi (merkeze 23 km) trenle gelenin de 1. gününe yazardı ve REPORT ölçümlerini kaydırırdı.
Ölçülen etki (PAR, 3 gün, 12 saat): 1. gün CDG seçilince 10 durak / 3.84 km yerine
8 durak / 26.19 km — transfer gerçekten bütçeden yiyor. Gün dict'i `starts_from` taşıyor,
UI 1. güne "Starts from ..." başlığı basıyor.

**Aşama 2 (2026-08-27): OTP kurulmadı, RAPTOR yazıldı.** Karar öncesi karşılaştırma yapıldı;
sonuç şu: **OTP2'nin transit yönlendiricisi zaten RAPTOR**, ama API'si yolcu bilgilendirmesi için
tasarlandığından tek-noktadan-çoğa sorguyu ifade edemiyor (OTP2 bunu analiz tarafına bıraktı;
kalan `SandboxAPITravelTime` isochrone/raster döndürüyor ve "desteklenmiyor" işaretli). Çift çift
sorulunca Paris'in 379 katalog noktası 143.641 sorgu; RAPTOR ise her kalkış noktasını tek koşuda
bütün duraklara doldurduğu için 379 koşu. İhtiyacımız olan algoritmaydı, sunucu değil. Ayrıca
JDK 21 + docker-compose kurmak, #87'de OSRM'i atma gerekçesiyle çelişiyordu.

`optimization/raptor.py` (Delling/Pajor/Werneck 2012) + `pipeline/build_transit_matrix.py`.
IDFM feed'i (135 MB zip, açılınca 1.3 GB; `stop_times.txt` tek başına 992 MB / 11.4M satır)
zip'ten akıtılarak okunuyor, diske açılmıyor. Tek servis günü: 2.02M çağrı, 99.342 sefer,
2.430 desen. Bütün kalkış noktaları **tek seferde** çözülüyor — varışlar (kalkış × durak) dizisi,
her adım tek numpy işlemi.

Sessiz-yanlış-cevap tuzakları ve çözümleri: **(a)** feed'in ilk üç günü (24-26 Ağu) yayım
tarihinden önce olduğu için neredeyse boş — "ilk Çarşamba" sezgisi 5.027 seferlik kütük bir güne
düşüyordu, artık span içindeki **en dolu** Çarşamba seçiliyor (144.374 sefer); **(b)** aynı durak
dizisinde birbirini geçen seferler — RAPTOR ilk kalkanı bindirdiği için ekspres kaçıyordu, kurma
anında bölünüyor (Paris'te **30 desen**); **(c)** GTFS'in `24:40:00` yazımı — kırpılsa bütün gece
hatları kaybolurdu, saniye olarak saklanıyor. Üçü de teste bağlandı.

Matris tek bir saate sabitlenmesin diye **08:00-20:00 arası 13 kalkıştan medyan** alınıyor.
Erişim ve aktarma yürüyüşleri Aşama 1'in gerçek yürüme ağından geliyor, kuş uçuşu değil. Her çift
ayrıca doğrudan yürüyüşle karşılaştırılıp hızlısı tutuluyor — yani transit modu zaten "yakınsa
yürü, uzaksa bin" davranışı, ayrı bir hibrit moda gerek yok.

**Sonuçlar:** çiftlerin %97'sinde transit yürümeyi yeniyor, medyan 2.07 kat; ortalama yolculuk
25 dk (her yeri yürümek 55 dk); efektif kapıdan-kapıya kuş uçuşu hız medyan 7.8 km/s.
**CDG → Notre-Dame: 5.8 saatlik yürüyüş yerine 50 dakika.** Elle doğrulanmış yolculuklarla
tutuyor (Gare de Lyon → Louvre 15 dk, Eyfel → Louvre 30 dk, Gare du Nord → Sacré-Cœur 15 dk).
GTFS feed'leri hızlı bayatlıyor (IDFM yalnızca 2026-08-24..09-25'i kapsıyor), yani periyodik
yenileme gerekiyor.

**Berlin de eklendi (2026-08-27), ama önce bir sessiz-yanlış-cevap daha çıktı.** VBB feed'i
(79 MB zip, 670 MB açık) aynı script'e bir satırdı; ilk koşu **havalimanından şehre 240 dakika**
verdi (gerçek ~50). Sebep router'da değil veride: `transport.csv`'deki havalimanı koordinatı
**pistlerin ortası**, en yakın yaya yolu düğümü 1.225 m batıda, Waßmannsdorf köyünde. Erişim ağ
üzerinden oradan ölçülünce yolcu köy otobüsüne biniyordu; havalimanının kendi peronu kuş uçuşu
785 m ötedeydi ve hiç değerlendirilmiyordu. Üç düzeltme:

- **Snap güveni** (`SNAP_TRUST_METRES = 150`): nokta yaya ağına yakın oturuyorsa ağ mesafesi,
  oturmuyorsa kuş uçuşu × 1.3. Yanlış düğümden ölçülen ağ yolu "biraz hatalı" değil, başka bir
  sorunun cevabı. Paris'te 14.147 durağın 14.130'u, Berlin'de 6.412'nin 6.386'sı ağa oturuyor.
- **`parent_station` bağlantıları**: GTFS istasyon peronlarını zaten gruplar, biz kullanmıyorduk.
  Flughafen BER'in 21 peronu tek ebeveyni paylaşıyor ve terminal içi yürüyüşü hiçbir yaya ağı
  tarif etmiyor. Paris 54.032, Berlin 3.322 peron bağlantısı.
- **Yarıçap ile maliyeti ayır**: detour çarpanı yürüyüşün modeli, "ne kadar yürünür"ün değil.
  İkisine birden uygulayınca 785 m'lik peron 1.020 m'ye şişip 800 m yarıçapın dışında kalıyordu.

Bu düzeltmeler Paris'i de iyileştirdi (yeniden üretildi): %97 → **%98**, 2.07 → **2.14 kat**,
ortalama 25 → **23 dk**. Berlin: %95, 1.98 kat, ortalama 27 dk (yürüme 56). Doğrulanan yolculuklar
— **havalimanı → Brandenburg Kapısı 45 dk**, → Alexanderplatz 53 dk, Hauptbahnhof →
Alexanderplatz 13 dk, Brandenburg Kapısı → Reichstag 4 dk (yürüme berabere). Dosyalar
`PAR_transit.npz` 0.8 MB + `BER_transit.npz` 0.4 MB. Erişimsiz kalan tek yer Tempelhofer Feld:
merkezi en yakın duraktan 956 m, ki bu doğru — o bir park.

**Uzak hub uyarısı (2026-08-27).** Parça 1 ile Aşama 2 birleşince ortaya çıkan boşluk: kullanıcı
havalimanı + "On foot" seçerse hâlâ 5.8 saatlik transfer alıyordu. İtinerary bunu zaten dürüstçe
gösteriyor (1. gün 23.84 km, iki durak eksik) ama ancak planlamadan *sonra* ve ancak 2. günle
karşılaştıran birine. `views/itinerary._arrival_transfer_hint` seçim anında uyarıyor: transfer
60 dakikayı aşıyorsa **ve** transit en az 1.5 kat hızlıysa. Rakamlar router'ın kullanacağı
`_build_distance_functions` ile hesaplanıyor, ayrı bir kestirimle değil — uyarı uyardığı
itinerary'yle çelişemesin diye. Bilinçli olarak **engellemiyor ve otomatik değiştirmiyor**:
Orly'den yürümek tuhaf bir tercih, geçersiz değil. Ölçülen eşik davranışı: CDG + yürüme uyarıyor
(5s 40dk → 51 dk), CDG + araba uyarmıyor (65 dk → 50 dk, kesintiye değmez), merkezi garlar
uyarmıyor, Berlin uyarmıyor (önerecek transit yok).
[#32](https://github.com/yukselburcinn-web/DA592/issues/32)

---

## Öncelik sırası (2026-08-26 akşamı itibarıyla açık işler)

| Öncelik | Issue | Neden |
|---|---|---|
| 1 | #71 | Rota kalitesi için kalan en yüksek getirili veri işi: router tamamen zaman pencerelerine dayanıyor ve pencerelerin %60'ı kategori varsayımı. Aynı issue bütçe slider'ını da kapsıyor |
| 2 | #7 | `priority:high` etiketli, final rapor için gerçek LLM sonucu eksik |
| 3 | #76 | Gerçek hata: #70'in düzelttiği şey LangGraph yolunda hâlâ bozuk, ve #72'den sonra bedeli büyüdü |
| 4 | #49, #77 | Ölçüm dürüstlüğü, aynı hata şekli iki yerde: ikisi de tavansız bir oranı 1.0 üzerinden okuyor. Birlikte ele alınmalı |
| 5 | #79 | Altı slider'ın biri hiçbir şey yapmıyor; taksonomi kararı, retrieval'a dokunuyor |
| 6 | #30, #33 | Veri kalitesi/granülerlik ince ayarı. #33 ayrıca #72'nin kalabalık çarpanının ön koşulu |
| 7 | #32 | Kapsamlı altyapı işi. Parça (1) küçük ve bugün yapılabilir; Aşama 2 teslim penceresine sığmaz |
| 8 | #78, #80, #81 | #72'nin bıraktığı iyileştirme ve temizlik işleri; hiçbiri bugünkü kaliteyi engellemiyor |
