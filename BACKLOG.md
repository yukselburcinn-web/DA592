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

## Durum özeti (2026-08-26)

| Alan | Açık | Kapalı |
|---|---|---|
| A — Veri & Modeller | 4 (#30, #33, #70, #71) | 5 (#1, #2, #3, #27, #65) |
| B — Retrieval & Bilgi Grafiği | 1 (#49) | 8 (#4, #5, #6, #42, #46, #48, #50, #63) |
| C — Ajanlar, Orkestrasyon & UI | 2 (#7, #72) | 16 (#8, #9, #10, #19, #20, #21, #22, #23, #29, #41, #54, #56, #57, #59, #61, #67) |
| D — Altyapı | 1 (#32) | 4 (#11, #12, #26, #31) |

İlk 12 maddelik listenin tamamı kapalı — proje ilk backlog'u bitirdi. Açık kalan işlerin hepsi ilk
listede yoktu; uygulama canlı test edilirken ya da kod incelenirken çıkan bulgular.

Toplam 41 issue: **8 açık, 33 kapalı.** Sayılar ve durumlar 2026-08-26 itibarıyla `gh issue list`
ile doğrulandı.

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

### #70. Açılış saatleri parse'ı OSM etiketinin %92'sinde bilgi kaybediyor — 🔓 Açık
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

[#70](https://github.com/yukselburcinn-web/DA592/issues/70)

### #71. Google Places ile zenginleştirmeyi değerlendir — 🔓 Açık (karar issue'su)
*(enhancement, `priority:low`)* Uygulama değil **karar** issue'su. #32 bunu zaten gerekçesiyle
reddetmişti: Places içeriğinin kalıcı önbelleklenmesi ToS ile yasak, `place_id` dışında istisna yok.
Engel bu proje için somut — `poi.csv` public depoda commit'li.

Üç seçenek gövdede: (A) sadece `place_id` + çalışma anı çekme — offline ve **belirlenimcilik**
kaybı, not verme tekrarlanamaz hâle gelir; (B) izin veren kaynaklar (Overture CDLA-Permissive,
Foursquare OS Places Apache-2.0, OSM ODbL); (C) reddi teyit et.

**#70 çözülmeden karar verilemez** — istenen zenginleştirmenin çoğu doğrulanmış açılış saati ve o
veri zaten elimizde. #70 sonrası Google'ın gerçek marjinal katkısı ölçülebilir.

[#71](https://github.com/yukselburcinn-web/DA592/issues/71)

### #33. Talep tahminine ve fiyat sinyaline şehir düzeyinde granülerlik ekle — 🔓 Açık
*(enhancement, `priority:medium`)* Talep verisi Eurostat `tour_occ_nim`'den geliyor: gerçek, aylık,
COVID çöküşünü içeren bir seri — ama granülerliği **ülke düzeyinde**. Yani Paris'in talep tahmini
tüm Fransa'nın turist sayısını proxy alıyor. REPORT §5'in kendi ifadesiyle "tek dürüst kalan boşluk".
Eurostat'ın şehir serisi (`urb_ctour`) var ama **yıllık**, aylık mevsimsellik varsayan Holt-Winters'ı
besleyemiyor.

İkinci yarısı fiyat: `price_level` ile crowding tahmini hiçbir yerde birbirine bağlanmıyor, yani
proposal'ın "bütçeyi mevsimsel talebe göre hizala" çerçevesinin sadece crowding tarafı gerçek veriyle
karşılanıyor. Önerilen kaynak: Inside Airbnb. [#33](https://github.com/yukselburcinn-web/DA592/issues/33)

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

### #72. Rota modelini TSP'den TOPTW'ye taşı — 🔓 Açık
*(enhancement, `priority:medium`)* Router bir TSP çözüyor: hepsini sırala, mesafeyi küçült, sığmayanı
ele. Turistin problemi ise **seçim** — hangi duraklar. Doğru formülasyon **TOPTW** (Team Orienteering
Problem with Time Windows): puanlı düğümler, günlük zaman penceresi, *m* gün için *m* tur, puanı
maksimize et.

Bugünkü yamalar aynı eksikliğin izleri: `_fill_days_to_budget`, `_rebalance_days`,
`_ensure_daily_meals`, `_ensure_evening_stops`, `NIGHTLIFE_EARLIEST_HOUR`, `_nightlife_last` —
hepsi TOPTW'de tek bir modelin kısıtı. Ve birbirinden habersiz olmalarının **ölçülmüş bedeli var**:
aynı aday setinde `min_food_per_day=0` → 3 nightlife durağı, `=2` → 2 durak; öğün geçişi akşam
geçişinin koyduğu barı düşürüyor. 2 öğün garantisi de tutmuyor.

Kazanç ölçüldü ama **bedelsiz değil**: 72 günde greedy zaman-pencereli yerleştirme durak/gün'ü
7.76 → 8.24 (+%6) çıkarıyor, ama **km/durak 1.04 → 1.16 (+%12)** — bu, karşılaştırmalı analizde
*itinerary coherence* olarak raporlanan metrik. Greedy mesafeyi optimize etmiyor; gerçek çözücünün
(OR-Tools routing/CP-SAT, ya da Vansteenwegen'in ILS'i) bunu km cezası olmadan alması bekleniyor —
**doğrulanmadı, issue'nun ilk işi bunu ölçmek.**

#32 bağımlılık değil: TOPTW haversine matrisiyle de çalışır. #70 ise ön koşul — zaman pencerelerinin
gerçek veriye dayanması için.

[#72](https://github.com/yukselburcinn-web/DA592/issues/72) · [PR #62](https://github.com/yukselburcinn-web/DA592/pull/62)

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

## Öncelik sırası (2026-08-26 itibarıyla açık işler)

| Öncelik | Issue | Neden |
|---|---|---|
| 1 | #70 | Rota kalitesi için en yüksek getirili veri işi: haftanın günü hiç taşınmıyor, pazartesi kapalı müze pazartesiye konabiliyor. Yeni kaynak gerektirmiyor, #72'nin de ön koşulu |
| 2 | #7 | `priority:high` etiketli, final rapor için gerçek LLM sonucu eksik |
| 3 | #49 | Ölçüm dürüstlüğü: recall tavanı 1.0 değil ~0.573, ama 1.0 üzerinden okunuyor |
| 4 | #72 | TOPTW'ye geçiş — altı özel-durum yamasını tek modele indirir; km/durak takası önce ölçülmeli |
| 5 | #30, #33 | Veri kalitesi/granülerlik ince ayarı |
| 6 | #71 | Karar issue'su, #70 tamamlanmadan cevaplanamaz |
| 7 | #32 | Kapsamlı altyapı işi — Aşama 1 (Valhalla/GraphHopper) tek başına da değerli, Aşama 2 (transit) stretch goal |
