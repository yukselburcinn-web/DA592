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

## Durum özeti (2026-08-28)

| Alan | Açık | Kapalı |
|---|---|---|
| A — Veri & Modeller | 1 (#92) | 10 (#1, #2, #3, #27, #30, #33, #65, #70, #71, #79) |
| B — Retrieval & Bilgi Grafiği | — | 11 (#4, #5, #6, #42, #46, #48, #49, #50, #63, #85, #113) |
| C — Ajanlar, Orkestrasyon & UI | 2 (#7, #94) | 24 (#8, #9, #10, #19, #20, #21, #22, #23, #29, #41, #54, #56, #57, #59, #61, #67, #72, #76, #77, #78, #80, #81, #83, #109) |
| D — Altyapı | — | 7 (#11, #12, #26, #31, #32, #93, #101) |

İlk 12 maddelik listenin tamamı kapalı — proje ilk backlog'u bitirdi. Açık kalan işlerin hepsi ilk
listede yoktu; uygulama canlı test edilirken ya da kod incelenirken çıkan bulgular.

Toplam 55 issue: **3 açık, 52 kapalı.** Sayılar ve durumlar 2026-08-28 itibarıyla
`gh issue list` ile doğrulandı ve her issue'nun aşağıda kendi bölümü var.

**Bu güncellemede ne değişti:** #32 üç parçasıyla kapandı (self-hosted sokak mesafesi, GTFS
transit, varış hub'ı), #81 kapandı (`POIZoner` kaldırıldı) ve #30 kapandı — ölçüldüğü 47 ikonluk
sekiz şehir listesi iki şehirlik katalog geçişiyle emekli oldu. #32'nin kapsam dışı bıraktıkları
#93'e taşındı ve #93 aynı gün kapandı: 3. maddesi (`use_real_routing` varsayılanı) #94'e
taşındı, 1. ve 2. maddeleri (tek iş günü matrisi, GTFS tazeleme) aksiyon alınmadan kapatıldı.
#92 ve #94 uygulama canlı sürülürken çıktı. #49 da kapandı: recall'ın yapısal tavanı
çözülmedi ama artık hem REPORT'ta hem uygulamada yazılı, yani gizli değil. #72'nin bıraktığı iki
`priority:low` iş de kapandı: #78 kapsam dışı bırakıldı, #80'in ölçülmüş temeli ise yeniden ölçüldü
ve tutmadı. Son üç kapanış: **#101** (Quickstart artık veri üretmiyor, REPORT'un kendi içindeki
çelişkileri giderildi — PR #104), **#71** (Google Maps kazıma, veri commit edilmeden; kalan %5,0
yanlışlık REPORT §5'te yazılı) ve **#76** (LangGraph yolu da `start_date`'i router'a ve forecaster'a
geçiriyor; iki davranış testiyle korunuyor — PR #115) ve **#77** (router artık durak sayısını
ulaşılabilir tavana karşı raporluyor — PR #117). Bununla birlikte **B ve D alanlarında açık iş
kalmadı**: geriye kalan üçün ikisi C'de (#7, #94), biri A'da (#92). Ayrıca daha önce backlog'a hiç
yazılmamış iki kapalı issue (#83, #85) eklendi. Önceki güncellemede: #70 ve #72 kapandı. #72 (TOPTW router) altı yeni bulgu doğurdu —
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

### #30. POI kataloğunda hâlâ eksik olan 5 ikonik yer — ✅ Kapalı
*(Etiket: `area:agents`, içerik: veri/kategori kotası — etiket içerikle örtüşmüyordu. 2026-08-26'da
kapatıldı.)* #28'in şöhret-bazlı sıralamasının ardından ikon kapsaması 20/47'den 42/47'ye çıkmış,
Kapalıçarşı, Sacré-Cœur, Foro Romano, Alfama ve Praça do Comércio dışarıda kalmıştı.

**Sorunun beşte dördü sorulmayı bıraktı, biri gerçekten girdi.** İki şehirlik katalog geçişi
(`Replace the eight-city catalogue with a two-city one`) 47 ikonluk sekiz şehir listesini geçersiz
kıldı: Kapalıçarşı (İstanbul), Foro Romano (Roma), Alfama ve Praça do Comércio (Lizbon) artık
kapsanan şehirlerde değil. Kalan tek aday **Sacré-Cœur katalogda** (`POI0017`, PAR, `religion`,
`sitelink_count` 59). Bugünkü katalog: **PAR 371 + BER 283 = 654 POI.**

Kota esnekliği ve `place=square` gibi eksik etiketler ayrı bir soru olarak duruyor, ama artık bu
issue'nun gövdesindeki ölçüye bağlı değil — kapsama iddiası ölçüldüğü listeyle birlikte emekli oldu.
[#30](https://github.com/yukselburcinn-web/DA592/issues/30) — ilişkili: #28 (merge edilmiş), #2

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

### #71. Açılış saati ve fiyat kapsamını kapat: kaynak kararı — ✅ Kapalı
*(enhancement, `priority:high` — 2026-08-26'da yeniden yazıldı, 2026-08-28'de ölçüldü ve karara
bağlandı)* Karar issue'su, uygulama değil.

**Sorun.** Kataloğun 654 POI'sinde açılış saati 260'ında (%39,8) gerçek OSM etiketi, gerisi kategori
varsayımı; fiyat 80'inde (%12,2) gerçek. TOPTW tamamen zaman pencerelerine dayandığı için router'ın
kısıtlarının %60'ı tahmin, ve bütçe slider'ı hiçbir şey ifade edemiyor (`price_level` tek bit, 61
food POI'nin hepsi aynı değerde).

**Karar: Google Maps kazıma, veri commit edilmeden.** Kazıma yapıldı ve ölçüldü; çıktı `local/`
altında ve `.gitignore`'da. `poi.csv` bir satır değişmedi, `hours_source`/`price_source` yalnızca
`osm` ve `category_default` demeye devam ediyor — çünkü commit'li satırlar gerçekten o. Kaynağın
şartları içeriğin yeniden dağıtımına izin vermiyor ve bu depo public; asıl engel baştan beri API
mekanizması değil, içeriğin nereye yazıldığıydı.

**Ölçüm (2026-08-28).** 654 POI'nin 511'i eşleşti. Doğrulanmış haftalık saat 326 (%49,8), puan+yorum
508 (%77,7), gün içi yoğunluk 269 (%41,1), fiyat kademesi 60 (%9,2). Katalogla birleşince açılış
saati **%39,8 → %59,9**; fiyat tek değerden dört kademeye (371/31/21/5), `food` dördüne birden
yayılıyor, `shopping` hiçbir şey kazanmıyor.

Router etkisi — 2 şehir × 4 arketip × 3 gün, graf/retrieval/havuz bir kez kurulup paylaşıldı, tek
değişken veri: **kapsam artışı planı büyütmüyor ya da derli toplu yapmıyor** (7,71 → 7,79 durak/gün;
0,514 → 0,510 km/durak). Değiştirdiği şey doğruluk: bugünkü katalog denetlenebilir 101 durağın
5'ini (**%5,0**) kapalı mekâna koyuyor, kategori varsayımı kullandığı duraklarda 32'de 3 (**%9,4**).
Ayrıntı REPORT §5'te ve [issue yorumunda](https://github.com/yukselburcinn-web/DA592/issues/71#issuecomment-5445581883).

**Overture / Foursquare ölçümü yapılmayacak (2026-08-28 kararı).** Issue'nun ilk hâli izin veren
kaynakların kapsamını ölçmeyi şart koşuyordu. O kriter düşürüldü: kazıma zaten yapıldı, bedeli
ölçüldü ve veri depoya girmiyor, dolayısıyla ölçüm artık hangi kaynağın seçileceğini değiştirmiyor —
teslim penceresinde karşılığı olmayan bir iş olurdu. Proje bugünkü kapsamla ship ediliyor ve %5,0 /
%9,4 rakamı REPORT'ta açıkça yazılı.

Kalan iş yok; issue kapatılmayı bekliyor.
**Kapatıldı (2026-08-27).** Karar verildi, uygulandı, bedeli ölçüldü ve kalan iş bırakılmadı.
Kapsam artışının router çıktısına etkisi yok (7,71 → 7,79 durak/gün; 0,514 → 0,510 km/durak);
değiştirdiği şey doğruluk — bugünkü katalog denetlenebilir 101 durağın 5'ini (%5,0) kapalı mekâna
koyuyor, kategori varsayımı kullanılan duraklarda 32'de 3 (%9,4). Bu rakam REPORT §5'te gizlenmek
yerine yazılı ve proje bu kapsamla ship ediliyor. **Overture / Foursquare ölçümü düşürüldü:** kalan
kapsamda izin veren kaynakların hiçbiri entegre edilmeyeceği için ölçüm ship edilen hiçbir şeyi
değiştirmezdi.
[#71](https://github.com/yukselburcinn-web/DA592/issues/71)

### #33. POI başına zamanla değişen kalabalık sinyali — 🔒 Kapalı
*(enhancement, `priority:medium`)* Issue'nun ilk gerekçesi — talep verisinin ülke düzeyinde olması —
`pipeline/build_demand.py`'nin NUTS 2'ye geçmesiyle kapandı. İkinci gerekçesi — `crowding_discount`'ın
sabit `1.0` döndürmesi — #72/#71 ile kapandı: `data/crowding.csv` POI başına saatlik seriyi getirdi.

**Üçüncüsü de kapandı: saat.** Puan statik bir düğüm ağırlığı olduğu için ziyaret saati ona girdi
olamıyordu; TOPTW'de zamana bağlı düğüm ödülü için de yer yok. Yerine ölçülen POI'ye günde **ikinci
bir düğüm** veriliyor: kendi en sakin üç saatine sabitli ve gün boyu kopyasından ucuz fiyatlanmış
(`toptw._crowd_slots`). Saati seçmek böylece düğüm seçmeye dönüşüyor. 2 şehir × 4 arketip × 3 gün
uzunluğunda ölçüldü (`evaluation/crowding_hour_measurement.py`): durakların kendi ortalamalarına
göre fazladan yoğunluğu **+11.1 puandan +2.0'ye**, maruz kalınan yoğunluk %58.6'dan %49.5'e indi;
bedeli durakların %0.4'ü ve durak başına %2.9 mesafe. 8 şehir-arketip çiftinin 7'sinde iyileşiyor.

Kalan iki madde de kapandı:
- **Kaynak sütunu** eklendi: `crowding.csv` artık `poi_id,day,hour,busy,source`.
- **Kapsam %41.1**, ölçülmüş bir sınır olarak kabul edildi. Ziyaret edilen duraklarda %56.5'e
  karşılık geliyor (router zaten merkezi POI'leri seçiyor); kalan %44 kategori ortalamasıyla
  fiyatlanıyor ve sakin pencere alamıyor, yani körlemesine zamanlanıyor. Kabul edilmesinin
  gerekçesi ölçüldü: ölçülmeyen kısım **yanlı bir örneklem değil** — popülerlik (3.7/3.7),
  sitelink (17/16) ve aylık pageviews (715/721) medyanlarında fark yok. Kategori ortalaması
  yedeğinin leave-one-out hatası medyan 6.5 puan (POI'ler arası yayılım 13.2), yani sinyalin
  kabaca yarısını kurtarıyor. Kazımanın tavanı zaten ~%45: 654 önbellek kaydının 293'ünde
  `popular_times` var. Kapsamı büyütmenin tek yolu ayrı bir kaynak (Wikipedia pageviews aylık
  serisi) ve o **aylık** olduğu için saat yarısına hiçbir şey katmaz — ayrı iş olarak ayrıldı.

[#33](https://github.com/yukselburcinn-web/DA592/issues/33) · [PR #108](https://github.com/yukselburcinn-web/DA592/pull/108)
— ilişkili: #71 (kaynak kararı), #72 (puan formülü), #109 (yemek oturumları), REPORT §5

### #79. Macera slider'ı puanı hiç hareket ettirmiyor — 🔒 Kapalı
*(enhancement, `priority:medium`)* #72'nin puan fonksiyonu, tercih vektörünü kategori ağırlıklarına
çeviren matrisi `user_survey.csv` ile `CATEGORY_AFFINITY` arasından NNLS ile türetiyor. Türetilen
matriste **`adventure` satırının tamamı sıfır**: kullanıcı slider'ı nereye çekerse çeksin puan
değişmiyor.

Sebep taksonomi: katalogda maceraya karşılık gelen kategori yok, en yakın aday `beach` ve iki
şehirlik sette `beach` kategorili **sıfır POI** var. Ankette `adventure` ile `nature` korele olduğu
için NNLS ortak varyansı tamamen `nature`'a veriyor.

**Karar: B — taksonomiye dokunmadan arayüzü dürüst yap.** A seçeneği (kataloğa `adventure`
kategorisi eklemek) veriyle denendi ve zemini zayıf çıktı: anahtar kelime taraması 61 POI buluyor
ama içinde Sorbonne ve Tour d'Argent gibi açık yanlışlar var, yani #65'in temizlediği sınıftan bir
taksonomi riski. Retrieval 10 kategoriye bağımlıyken teslime bir haftadan az kala alınacak risk
değil.

Yapılanlar:
- **Slider arayüzden kaldırıldı**, altıncı anket özelliğine anket ortalaması (0.42) besleniyor.
  KMeans hâlâ altı özellikle çalışıyor, `user_survey.csv` değişmedi, yedi arketibin hepsi kalan beş
  slider'dan ulaşılabilir durumda (Nature & Adventure profillerin %5.8'inden %2.5'ine düşüyor,
  `nature` üzerinden geliniyor). Kaldırma kararını belirleyen ölçüm: slider etkisiz olmakla
  kalmıyordu, KMeans arketibini değiştiriyordu — macerayı sonuna kadar açan kullanıcı **"Budget
  Backpacker"** olarak sınıflanıp alakasız bir sebeple farklı plan alıyordu.
- **`beach` fitten ve arketip sorgularından çıktı.** Matrisin %8.2'siydi (relax satırının %30'u,
  nature'ın %25'i) ve iki arketibin sorgusu denizi olmayan şehirlerde "beaches" istiyordu. Bu
  değişiklik **davranışı hiç değiştirmiyor** ve öyle iddia ediliyor: `preference_match` tek
  kategoriye bakıyor, `_MAX_MATCH`'i `food` sütunu belirliyor, aynı POI'ler seçiliyor. Kazanç
  matrisin ve sorguların elimizde olmayan bir kataloğu anlatmayı bırakması. Filtre katalogdan
  türetildiği için kendi kendini onarıyor: denizi olan bir şehir eklenirse `beach` fite geri döner.
- **Bütçe slider'ı hakkındaki iddia düzeltildi.** REPORT "#71'den sonra slider fiyata göre çalışıyor"
  diyordu; yanlış. `price_level` kodda yalnızca iki yerde okunuyor ve ikisi de puan değil:
  orchestrator'ın "ücretsiz oranı" istatistiği ve arayüzdeki "· free" etiketi. Slider puana yalnızca
  `shopping` (0.306) + `landmark` (0.095) üzerinden ulaşıyor — toplam 0.400, `culture`'ın 3.205'ine
  karşı — ve uçtan uca 24 POI'nin 2'sini değiştiriyor. Etiketi bu yüzden "Budget" değil "Everyday or
  upmarket".

Kapsam dışı kalan ve ayrı issue'ya taşınan bulgu: **retrieval `top_k=24`'te tek kategoriye yığılıyor**
— Culture Enthusiast havuzu 19 müze + 4 landmark + 1 history geliyor, `religion` ilk kez `top_k=120`'de
görünüyor. Kataloğun büyük bir kısmı bu yüzden ürüne ulaşmıyor.
[#79](https://github.com/yukselburcinn-web/DA592/issues/79)

### #92. Ulaşım hub'ları pist merkezinden konumlanıyor, terminalden değil — 🔓 Açık
*(bug, `area:data`, `priority:medium` — 2026-08-27)* `transport.csv` koordinatları OSM'den
`out center` ile alınıyor (`pipeline/build_transport.py`), yani bir havalimanı için **pistlerin
geometrik ortası**. Yolcunun bulunduğu yer orası değil.

**Nasıl görüldü:** #32 Aşama 2'de Berlin transit matrisi ilk üretildiğinde havalimanından merkeze
**240 dakika** çıktı; gerçeği ~45 dk. Hiçbir şey patlamadı — makul görünen bir sayı beş kat yanlıştı.
`Berlin Brandenburg Airport` koordinatı `52.365932, 13.49691`, en yakın yaya yolu düğümü **1.225 m
batıda, Waßmannsdorf köyünde**; erişim oradan ölçülünce yolcu köy otobüslerine biniyordu, oysa
havalimanının kendi peronları (`Flughafen BER`) 785 m ötedeydi.

Bedel üç yerde: transit erişimi (şu an "ağa 150 m'den uzaksa kuş uçuşu × 1.3" kuralıyla **telafi
ediliyor, düzeltilmiyor**), sokak matrislerinde snap payı (BER araç ağında 1.254 m, yürümede
1.225 m; CDG'de 185 m — orada şanslıyız), ve 1. günün pistlerin ortasından başlaması. Büyük
parklarda aynı sapma var (Tempelhofer Feld 863 m) ama orada centroid savunulabilir; havalimanında
değil, çünkü yolcunun bulunduğu nokta bellidir.

**Kabul kriterleri:**
- [ ] Havalimanı hub'ı `aeroway=terminal` → bağlı `railway=station` → centroid sırasıyla seçiliyor
- [ ] Her havalimanı hub'ı en yakın yaya düğümüne **150 m'den yakın** (`SNAP_TRUST_METRES` içinde)
- [ ] BER ve CDG transit erişimi kuş uçuşu telafisi olmadan ağ üzerinden çalışıyor
- [ ] `transport.csv` yenilendikten sonra **altı matris de yeniden kuruldu**
- [ ] CDG → Notre-Dame ~50 dk, BER → Brandenburg Kapısı ~45 dk elle doğrulanmış değerleriyle tutuyor

Tren garı ve otobüs terminallerindeki `railway=station` düğümü zaten doğru yerde; kontrol edilmeli
ama muhtemelen dokunulmayacak.
[#92](https://github.com/yukselburcinn-web/DA592/issues/92) — ilişkili: #32

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

### #85. Results sekmesinde sorgu metinleri görünmüyordu — ✅ Kapalı
*(enhancement, `area:retrieval`/`area:ui`)* Karşılaştırma "51 test queries" diyip her sorgunun
sonucunu gösteriyordu ama **sorgunun metni hiçbir sütunda yoktu**; okuyucu `query_id=14`'ün ne
sorduğunu göremiyordu. Bu teorik bir endişe değildi: #50'de gold etiketlerinin yanlış olduğu
(bir "rahat akşam" sorusunun `culture` ile notlandırılması) ancak sorgular tek tek okunarak fark
edilmişti — yani mimari tercihi kanıtlayan sekme, kendi denetimini imkânsız kılıyordu.
`run_comparative_analysis` satırlarına `query` alanı eklendi ve commit'li sonuç CSV'sine sütun
**yeniden ölçüm yapmadan** işlendi (metin `query_id`'nin deterministik fonksiyonu; yeniden koşmak
latency sayılarını sebepsiz oynatırdı). PR #86.
[#85](https://github.com/yukselburcinn-web/DA592/issues/85)

### #49. recall_at_k yapısal olarak tavanlı, ama 1.0 üzerinden okunuyor — ✅ Kapalı
*(enhancement, `priority:medium` — kapandı 2026-08-27)* #46'daki değerlendirmenin ikinci bulgusu
(ilki #48). `recall_at_k` k=8'de ölçülüyor ama medyan sorgunun gold set'i çok daha büyük, dolayısıyla
**kusursuz sıralayan bir retriever bile 1.0'a ulaşamaz**.

**Kapatma kararı: tavan çözülmedi, ama artık gizli değil.** İstenen iki seçenekten ikincisi ("tavan
raporda açıkça yazılmalı") #48'in Wikivoyage gold-key revizyonu (`pipeline/retrieval_gold.py`,
a8da444) ve #85'in sorgu tablosu ile yan etki olarak indi:
- REPORT `top_k=8` için ortalama ulaşılabilir tavanı **0.512** olarak yazıyor ve Fusion'ın 0.253'ünün
  "ulaşılabilirin %49'u" olduğunu söylüyor
- Uygulamada sorgu tablosunun "Answers" sütunu gold set büyüklüğünü gösteriyor, yardım metni
  *"Only eight are retrieved, so a key larger than that caps recall below 1.0"* diyor
- `comparative_analysis_results.csv` `gold_size` sütunu taşıyor — tavan sorgu bazında hesaplanabilir

**Bugünkü ölçüm (61 sorgu, `top_k=8`)** — issue'nun gövdesindeki 18 sorgu / gold 13.6 / tavan 0.573
rakamları bayat: gold set ortalama **23.0** (medyan 17, maks 85), 61 sorgunun **53'ünde** gold > 8,
ortalama tavan **0.512**. Fusion 6 sorguda tam tavana oturuyor — kusursuz retrieval, düşük recall
olarak okunuyor. Yani tavan gold set büyüdüğü için biraz daha sıkı.

**Kapsanmadan kalan, RAG çalışmasına bırakıldı:** `views/system_logs.py`'deki metrik açıklaması hâlâ
1.0 üzerinden okutuyor ("Share of the graph-verified answer set the retriever actually surfaced") —
sorgu tablosu tavanı söylüyor, metrik başlığı söylemiyor. Normalize recall'ın ikinci metrik olarak
gösterilmesi ve nDCG/MAP geçişi de yapılmadı. Normalize edilecekse toplulaştırma seçilmeli:
oranların ortalaması (%56.6) ile ortalamaların oranı (%49) aynı şey değil, REPORT ikincisini kullanıyor.
[#49](https://github.com/yukselburcinn-web/DA592/issues/49) — ilişkili: #46, #48, #85, #77 (açık)

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

### #83. Harita legend'ı 3. günden itibaren kırpılıyordu — ✅ Kapalı
*(bug, `area:ui`)* Legend yatay ve **çizim alanının içinde**, sol alt köşedeydi — hem harita yarım
genişlikte bir sütunda (~291 px'e kadar iniyor) sığmıyordu, hem de OpenStreetMap atıf çubuğuyla
aynı şeridi paylaşıyordu. 3 günlük planda Day 3 hiç görünmüyordu, oysa haritada üç ayrı renkte rota
vardı. Legend haritanın **altındaki margin'e** alındı ve yüksekliği gün sayısından hesaplanıyor
(`_MAP_LEGEND_PER_ROW`, `_MAP_LEGEND_ROW_PX`), böylece rotanın üzerini kapatmıyor ve 1–5 gün
aralığının tamamında her günün anahtarı görünüyor. PR #84 (önce üste alındı, sonra alta taşındı).
[#83](https://github.com/yukselburcinn-web/DA592/issues/83) — ilişkili: #21

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
parçası o gün karşılanamıyordu — forecaster şehir-ay düzeyinde tek skaler döndürüyor, sabit çarpan
hiçbir seçimi değiştiremez. İkisi de sonradan kapandı: POI yarısı #71'in serisiyle, saat yarısı
#33'ün ikinci düğümüyle. `tahmin` parametresi hâlâ okunmuyor ve gerekçesi aynı.

[#72](https://github.com/yukselburcinn-web/DA592/issues/72) · [PR #75](https://github.com/yukselburcinn-web/DA592/pull/75)

---

### #76. LangGraph orchestrator router'a `start_date` geçirmiyordu — ✅ Kapalı
*(bug, `priority:medium` — kapandı 2026-08-28, PR #115)* `orchestrator.py` geçiriyordu,
`orchestrator_langgraph.py` geçirmiyordu; `PlanState`'te alan bile yoktu. Sonucu: **#70'in kapattığı
hata bu yolda açıktı** — tarih olmadan `_opening_intervals` verbatim OSM etiketini haftanın gününe
göre çözemeyip kaba `open_hour`/`close_hour` çiftine düşüyor, yani `Tu-Su 10:00-18:00; Mo off`
"her gün 10-18 açık" diye okunuyordu. #72 router'ı tamamen pencere-bağımlı hale getirdiği için bedeli
yanlış bir durak değil, yanlış gün ataması ve yanlış saatti.

**Düzeltme, dördü de `orchestrator_langgraph.py`'de:** `PlanState`'e `start_date`, `plan_trip`
imzasına `start_date` (`orchestrator.py:101` ile aynı sırada), `travel_month` türetmesi `init_state`
kurulmadan önce, ve `_route`'ta router'a geçirme. Üçüncüsü salt parite detayı değil: `travel_month`
state'e girip forecast düğümü tarafından okunuyor, yani o satır olmadan tarih veren bir çağıran
yanlış açılış saatlerinin yanında **yanlış ay için kalabalık tahmini** de alıyordu.

**Testler.** Mevcut eşdeğerlik testi iki orkestratörün dönüş *şeklini* karşılaştırıyor, çıktısını
değil — hata bu yüzden ondan kaçtı; docstring'i ayrıca yanlış olarak "CI'da atlanır" diyordu, oysa
`tests.yml` `requirements-langgraph.txt`'i kuruyor. İki davranış testi eklendi ve **ikisinin de doğru
sebeple kırıldığı düzeltme geri alınarak doğrulandı**: `test_both_orchestrators_honour_the_weekday_a_start_date_names`
Centre Georges Pompidou ve Tour d'Argent'ı pazartesiye planlıyor (ikisi de o gün kapalı),
`test_a_start_date_also_reaches_the_forecaster_on_the_langgraph_path` tahmini 2026-09 yerine 2027-02
için alıyor. Yardımcı, pencereyi itinerary'nin kendi `date` alanına değil `start_date` + gün indeksine
göre çözüyor — tarihi hiç geçirmeyen orkestratör o alanı `None` bıraktığı için, alanı okumak testi
tam da yakalaması gereken itinerary'lerde atlamaya sokuyordu.

**REPORT §3.3 düzeltildi.** Cümle *"identical signature ... verified by
test_langgraph_orchestrator_matches_custom_orchestrator_interface"* diyordu ve iki yarısı da yanlıştı:
imza aynı değildi, ve o test imza eşitliğini zaten doğrulamıyor. Katalog ölçüsü issue yorumundaki
226/108/70 rakamlarıyla değil yeniden ölçülerek yazıldı (654 POI, her etiket yedi güne çözülüp kaba
çiftle karşılaştırılarak): **342 POI (%52,3)** en az bir günde farklı pencere veriyor, **153 (%23,4)**
kapalı olduğu bir günde açık okunuyor, pazartesi açık ara en kötü (**101**), sonra pazar (46).
Suite: 120 geçti, 1 atlandı.
[#76](https://github.com/yukselburcinn-web/DA592/issues/76)

### #77. Toplanan durak sayısını ulaşılabilir tavana karşı raporla — ✅ Kapalı
*(enhancement, `priority:medium` — kapandı 2026-08-28, PR #117)* #72'nin kapatılmamış kriteri.
Raporlanan durak sayısı ve km/durak **tavansızdı**: 9 durak iyi mi? Bu havuzda 11 mümkünse kötü.
#49'un retrieval tarafında tespit ettiği hatanın aynısı, ve #49 kapandığı için aynı ekranda biri
"ulaşılabilirin %49'u" diye okunurken diğerinin tavansız sayı vermesi tutarsızdı.

Ölçülecek büyüklük **puan değil durak**: çözücü her POI'ye sabit `DEFAULT_DROP_PENALTY_M = 8000`
uyguluyor, puan amaç fonksiyonuna hiç girmiyor — puan adayları seçer, çözücü seçilenin üzerinde
geometriyi tekdüze optimize eder.

**Ölçüm (yeniden alındı):** `evaluation/toptw_ceiling.py`, 48 konfigürasyon — shipped
**1178 / 1308 = %90,1**, ortanca %90,9, gezi başına 2,71 durak kaçıyor, tavana oturan 9 ve dokuzu da
`top_k=24`'te. Önceki %92,0 bugünkü kataloğdan ve #113 / #33 / #109 öncesindendi.

**Üçüncü kol: aradığı ayrışma yok, ve cevap bu.** Gezi kısa listeden, tavan havuzdan çözülüyordu;
aynı kısa listeyi serbest seyahatle çözen kol eklendi (kısa liste router'ın kendi `_select`'iyle
kuruldu). İki tavan 48 konfigürasyonun **35'inde birebir aynı**, kalan 13'te en fazla 2 durak ve
**iki yönde birden** oynuyor — 7'sinde kısa liste tavanı havuz tavanını *aşıyor*, ki alt küme
süperkümesini gerçek bir sınırda geçemez. Yani gevşetilmiş kol optimum değil, sabit
`SOLUTION_LIMIT = 150` altında sezgisel; büyük havuz daha zor arama. Kısa listenin maliyeti bu
çözünürlüğün altında (**−4 durak / 48 konfigürasyon**). Sonuç bölünmeden güçlü: **boşluğun tamamı
pratikte geometri**, kişiselleştirme gezi uzunluğundan ölçülebilir bir şey götürmüyor. "Kısa
listenin payı −%3" gibi bir yüzde bilerek basılmadı — gürültüyü bulgu gibi giydirmek olurdu.

**Yüzeye çıkarma iki katmanlı, çünkü maliyeti ölçüldü.** `RouterAgent.run(with_ceiling=True)`
`ceiling_stops` + `stops_ratio` döndürüyor ve `facts`'e bir cümle ekliyor, ama **varsayılan kapalı**:
mesafe aramayı budamadığı için gevşetilmiş çözüm annote ettiği geziden pahalı (Paris gezisi 9,65s,
tavanı üstüne 16,5s). Her zaman görünen sürüm Sistem günlükleri → Results sekmesinde, commit'li
ölçümden — retrieval'ın tavanı da zaten öyle raporlanıyor.

Yan bulgu: eski docstring tavanı "gerçek üst sınır" diyordu, üçüncü kol bunun doğru olmadığını
gösterdi; docstring ve arayüz metni düzeltildi. **Karşılanmayan tek kriter:**
`toptw_measurement.py`'ye ayrı sütun eklenmedi — o dosya "arm" tabanlı ve `toptw_ceiling.csv` zaten
o çıktının kendisi, oraya bir `ceiling` arm'ı koymak aynı pahalı çözümü ikinci kez koşmak olurdu.
[#77](https://github.com/yukselburcinn-web/DA592/issues/77)

### #78. Kilitle-ve-yeniden-çöz: "bu durağı değiştir" — ✅ Kapalı
*(enhancement, `priority:low` — kapandı 2026-08-27, kapsam dışı)* #72'nin kapatılmamış kriteri ve
kısıt modelinin doğal kazancı: bir durağı kilitlemek o POI'nin gün kopyasının `ActiveVar`'ını 1'e
sabitlemek, reddetmek tüm kopyalarını 0'a sabitlemek — ikisi de tek satır.

**Karar: model destekliyor, akış yazılmayacak.** Issue'nun kendi cümlesi ("eksik olan model değil,
arayüz ve akış") işin büyük yarısını tarif ediyor ve o yarı tek satır değil: uygulamada bugün
**hiç `session_state` yok** (`app.py`, `views/itinerary.py`, `views/system_logs.py` — 0 kullanım),
akış tek yönlü (slider → *Plan my trip* → render). Mevcut itinerary'yi + kullanıcı kararını taşımak,
olmayan bir kavramı mimariye sokmak demek. Üstüne iki şey daha biniyor: havuz daraldıkça öğün/doluluk
kısıtlarının infeasible olması (sessizce gevşemek yerine söylemek gerekir — ayrı bir hata yolu) ve
"aynı kilit kümesi = aynı çıktı" garantisinin REPORT'un determinizm iddiasına yeni test yükü koyması.

Rapor tarafında kayıp yok: #78 yeni bir yetenek değil, var olan yeteneğin arayüz imkânı. Başlanacaksa
doğru sıra — önce `toptw.solve`'a `locked`/`excluded` parametreleri (UI'sız, test edilebilir), sonra
oturum durumu, en son buton.
[#78](https://github.com/yukselburcinn-web/DA592/issues/78)

### #80. TOPTW formülasyonunun ~120 POI tavanı — ✅ Kapalı
*(enhancement, `priority:low` — kapandı 2026-08-28, ölçüm tekrar üretilemedi)* Router adayları
çözücüye vermeden önce puanla eliyor (`MAX_WORKING_SET = 120`) ve issue bunu bir tercih değil
zorunluluk olarak yazıyordu: 118 POI ~2 s'de çözülüyor, **371 POI (tüm Paris kataloğu) 10 dakikada
çözülmedi**.

**Yapısal tespit doğru ve duruyor.** Düğüm sayısı havuzla değil `havuz × gün` ile büyüyor: her POI
her gün için ayrı bir düğüm alıyor (o günün açılış penceresini taşısın diye), yemek POI'leri her gün
her öğün oturumu için bir kopya, ve aynı POI'nin bir kez girmesini bütün kopyaları kapsayan tek
`AddDisjunction` sağlıyor. `düğüm = (gezilecek × gün) + (yemek × gün × öğün) + 2`; Paris'te 371 POI /
3 gün / 2 öğün = **1.256 düğüm**. Üstüne çözümden *önce* doldurulan iki `düğüm × düğüm` matris
(~3,2 M hücre, karesel).

**Ama tavan iddiası tutmuyor (2026-08-28 ölçümü).** Ön filtre tamamen atlanıp tam katalog doğrudan
çözücüye verildi (`use_real_routing=False`, yürüme, 480 dk):

| şehir | havuz | gün | düğüm | süre | duraklar |
|---|---|---|---|---|---|
| Paris | 371 | 3 | 1.256 | **15,6 sn** | 6 · 8 · 8 |
| Paris | 371 | 4 | 1.674 | **20,5 sn** | 6 · 9 · 8 · 6 |
| Berlin | 283 | 3 | 893 | **9,2 sn** | 8 · 8 · 8 |
| Berlin | 283 | 4 | 1.190 | **8,2 sn** | 7 · 7 · 7 · 7 |

Ölçeklendirme (Paris, 3 gün, öğünsüz): 40 → 2,0 sn · 80 → 3,1 · 120 → 4,7 · 160 → 8,1 · 220 → 7,6 ·
371 → 15,9. Monoton bile değil, çünkü `SOLUTION_LIMIT = 150` sabit bir yineleme bütçesi.

**Sebebin bir kısmı:** issue'nun açıldığı gün (26 Ağu 15:59) aynı gün `f34de0e`,
`use_extended_swap_active`'i `len(pois) < gün × 9` koşuluna bağladı; ölçüm yenilenmedi. O ayar
zorlanınca 371 POI 15,9 → **45,2 sn** oluyor (commit'in kendi "4-5 kat" notuyla tutarlı). 45 sn de
10 dk değil — orijinal koşullar yeniden kurulamadı, iddia dar tutuldu: bugünkü kodla tavan orada değil.

**Asıl sebep, kapatmanın gerçek gerekçesi:** issue "çözücü ölçeklenmiyor, *bu yüzden* mecburen ön
filtre" diye çerçeveliyordu. Ön filtre `select_by_score` ve orası gezginin **altı slider'ının
itinerary'ye ulaştığı tek yer**; çözücü tercih vektörünü görmüyor (REPORT: çözücüyü aynı puanlarla
ağırlıklandırmak durak ve mesafe kaybettiriyor). Yani ön filtreyi kaldırmak, çözücü ne kadar hızlı
olursa olsun kişiselleştirmeyi kaldırmak demek — ölçüm de destekliyor: havuz 40 → 371'e çıkarken
durak sayısı iyileşmiyor (20 → 15 → 22).

**Kapanışta kalan iş:** REPORT §5 hâlâ eski cümleyi taşıyor — *"a full 371-POI city catalogue did not
solve in ten minutes"*. Tekrar üretilemeyen bir ölçüm; issue kapansa da düzeltilmeli.
`MAX_WORKING_SET` yükseltilebilir (220 POI 7,6 sn) ama geziyi iyileştirdiğine dair kanıt yok.
[#80](https://github.com/yukselburcinn-web/DA592/issues/80)

### #94. Harita kuş uçuşu çiziyor: duraklar arası düz doğru — 🔓 Açık
*(bug, `area:ui`, `priority:medium` — 2026-08-27)* `views/itinerary.py` haritayı
`go.Scattermap(mode="markers+lines+text")` ile çiziyor ve `lat`/`lon` olarak **doğrudan durak
koordinatlarını** veriyor. Duraklar arası her çizgi düz bir doğru — seçilen ulaşım modundan ve
"Use real street routing" işaretinden bağımsız.

Çalışan uygulamada Plotly trace'leri okundu (Paris, 4 gün, toplu ulaşım): gün başına
`points=7`, yani durak sayısı kadar. Gerçek güzergâh çizilse gün başına yüzlerce nokta olurdu.
Çizilenle panelde yazan da tutmuyor:

| gün | haritada çizilen | panelde yazan |
|---|---|---|
| Gün 1 | 3.17 km | **6.85 km** |
| Gün 2 | 5.18 km | **7.32 km** |
| Gün 3 | 5.91 km | **9.47 km** |
| Gün 4 | 3.14 km | **5.74 km** |

**Yeni bir gerileme değil**, harita en baştan böyleydi. Ama #32 mesafeleri haversine'den gerçek yol
ağına taşıyınca (Paris'te 4 günlük gezi 12.5 → 19.6 km) metinle çizim arasındaki çelişki büyüdü:
kullanıcı 6.85 km okuyup 3.17 km'lik çizgi görüyor.

Moda göre ne mümkün: **yürüme/araba/hibrit + real routing açık** → düzeltilebilir, yeni veriye
gerek yok (`{CITY}_{foot,car}.npz` düğüm koordinatlarını ve kenar listesini taşıyor; scipy
`dijkstra(..., return_predecessors=True)` polyline verir). **Toplu ulaşım** → çizilemez, transit
matrisi yalnızca dakika saklıyor, RAPTOR'un hangi hatlardan geçtiği tutulmuyor ve GTFS
`shapes.txt` (IDFM 135 MB, VBB 182 MB) hiç ayrıştırılmadı. **Real routing kapalı** → düz çizgi
zaten modelin kendisi, ama harita bunu rota gibi göstermemeli.

**Kabul kriterleri:**
- [ ] Real routing açıkken sokak modlarında polyline uzunluğu günün `distance_km`'ine **%10 içinde**
      yakınsıyor (şu an ~%50 sapma)
- [ ] Transit ve real-routing-kapalı bacaklar kesikli çiziliyor, başlık bunu söylüyor
- [ ] Maliyet sınırda: bacak başına bir Dijkstra (4 günde ~28 bacak), kalkış başına toplulaştırılabilir
- [ ] Rota boşsa ya da güzergâh bulunamazsa düz çizgiye düşülüyor, çizim kaybolmuyor

Transit güzergâhını gerçekten çizmek ayrı ve büyük bir iş (RAPTOR yolculuk ayrıştırması + `shapes.txt`
geometrisi); bu issue onu kapsamıyor.

**`use_real_routing` varsayılanı (#93'ten devralındı, 2026-08-27).** Bayrağın kapalı olmasının
*eski* sebebi (public OSRM uptime'ı) #87 ile ortadan kalktı; kalan tek sebep **karşılaştırılabilirlik**
— REPORT'taki bütün ölçümler bayrak kapalıyken alındı. Açmanın bedeli sıfır (matris araması), kazancı
gerçek: Paris'te 4 günlük gezi düz çizgiyle 12.5 km görünürken gerçek ağda 19.6 km, yani günler fazla
doluydu. Buraya taşındı çünkü yukarıdaki kabul kriterleri gerçek güzergâh çizimini zaten *real routing
açıkken* şart koşuyor: varsayılan açılırsa haritanın doğru çizmesi varsayılan yol olur, kapalı kalırsa
düz çizgi + kesikli gösterim varsayılan yol olarak kalır.
- [ ] `evaluation/comparative_analysis.py` bayrak açıkken yeniden koşulsun
- [ ] Değişen sayılar REPORT'a işlensin
- [ ] Varsayılan açılsın, ya da açılmama gerekçesi güncellensin

[#94](https://github.com/yukselburcinn-web/DA592/issues/94) — ilişkili: #21, #32, #93 (kapalı)

### #81. POIZoner artık router yolunda değil — ✅ Kapalı
*(`priority:low` — 2026-08-27)* #72 ile gün ataması modelin kararı olunca `POIZoner` router
tarafından hiç çağrılmayan bir modüle döndü; onu içe aktaran tek yer kendi testleriydi.

**Karar: B — kaldırıldı.** Çağrılmayan kod bir sonraki okuyucuya hâlâ kullanımdaymış izlenimi
verir, ve bu izlenimin bedeli somuttu: #32'nin gövdesi router'ı "POIZoner + 2-opt" diye tarif
etmeye devam ediyordu. Proposal'ın "iki bağımsız KMeans" anlatısı bu silmeyle **kaybolmuyor**:
ikinci KMeans `pipeline/city_guide.py` içinde, şehir rehberlerinin alan yapısını türeten kümeleme
olarak çalışıyor — ve orası, POIZoner'ın aksine, gerçekten çağrılıyor.

Silinen: `models/segmentation.py::POIZoner` (`_balance` dahil), `__main__` demo bloğunun zoner
kısmı, artık kullanılmayan `import math`, ve iki test — `test_poi_zoning_covers_all_pois`,
`test_balanced_zoning_evens_out_day_zones`. Üçüncü test korundu ama seviyesi değişti:
`test_zoning_returns_every_day_it_was_asked_for` iddiasının hâlâ tutması gereken yarısı (#63'ün
çökmesi: havuzda gezilecek hiçbir şey yokken de gezi istenen gün sayısıyla dönmeli) artık uçtan
uca router'a karşı doğrulanıyor — `test_a_trip_gets_every_day_it_asked_for_even_from_a_food_only_pool`.
`build_multi_day_itinerary` testlerinden biri zoner'ı yalnızca POI havuzunu tekrar düzleştirmek
için çağırıyordu, doğrudan havuza çevrildi. `TravelerSegmenter` etkilenmedi.

REPORT §3.2'deki anlatım tarihsel nota indirgendi (#19'un kapasite kısıtlı atamasının ne yaptığı
ve #72'nin bunu neden devraldığı duruyor), `city_guide.py` docstring'indeki eskimiş "POIZoner
applies downstream" ifadesi düzeltildi. Suite: **103 geçti, 1 atlandı.**
[#81](https://github.com/yukselburcinn-web/DA592/issues/81)

### #113. Kataloğun bir kategorisi hiçbir gezgine ulaşamıyordu — 🔒 Kapalı
*(bug, `area:retrieval`, `priority:medium`)* #79 üzerinde çalışırken çıktı. Proje bugüne kadar hep
"plan neyi içeriyor" diye sordu; "katalog neyi sunabiliyor" sorusu hiç sorulmamıştı ve cevabı çok
daha kötüydü. Her (şehir × arketip) retrieval havuzunun **birleşimi** — yani herhangi bir gezginin
görebileceği her şey — 3 günlük gezide kataloğun %63.8'iydi, ama `religion` bu birleşimde **84
POI'den 1'iydi**. Görülemeyen 83'ün içinde iki şehrin en bilinen 24 yerinden dördü vardı:
Notre-Dame de Paris, Panthéon, Sacré-Cœur ve Berlin Cathedral.

**İki bağımsız hata, ikisi de düzeltilmesi gerekiyordu.**

*Graf sıralaması:* `archetype_preferred_pois` tercih edilen POI'leri tek bir afinite × belirginlik
listesinde sıralayıp `top_k`'da kesiyordu. Bu, güçlü kategoriler kesimi doldurabildiği sürece en
zayıf kategoriyi aç bırakıyor: Culture Enthusiast / Paris'te 241 POI sıralanıyor ve ilk `religion`
(ağırlık 0.6) **132. sırada** — 3 günlük gezinin çektiği 72'nin çok ötesinde. Artık her kategori
kendi sıralamasını koruyup birleşik listeye ağırlığının belirlediği hızda giriyor (`i / ağırlık`):
listenin başı hâlâ gezginin en çok istediği şey (Louvre, Eiffel, Père Lachaise, Palais Garnier,
**Notre-Dame** — 5. sırada), ama istediği her kategori payını alıyor. İlk 72'nin dağılımı
17/16/15/14/10. Her iki backend de (NetworkX ve neo4j) artık aynı Python sıralamasını kullanıyor;
Cypher'ın kendi `ORDER BY`/`LIMIT`'i vardı ve ikisi bu noktada birbirinden sapabiliyordu.

*Kelime retriever'ı:* tokenizer kök almıyor, yani sorgu yalnızca birebir içerdiği kelimeleri
eşleştiriyor. İfade "places of worship" diyordu; 654 POI belgesinde `worship` 4, `places` 5 kez
geçiyor, `church` ise **69**. Kategori BM25'e görünmezdi ve sebebi bir yazım detayı gibi okunuyor.

**Ölçüm** (`evaluation/retrieval_coverage.py`, artık repoda): ulaşılabilir `religion` 1 → **30**,
3 günlük gezide ünlü POI kapsamı **20/24 → 24/24**. Retrieval kalitesi projenin kendi ölçüleriyle
gerilemedi: recall@k 0.253 → 0.258, archetype precision 0.967 → 0.969.

**Bedeli gerçek ve aşırı temsil edilen kategorilerin kuyruğu:** `landmark` ulaşılabilirliği
%39.7'den %19.8'e, `museum` %68.5'ten %42.6'ya iniyor, toplam 417 → 403. Üç kategorideki derinlik,
hepsindeki genişlikle takas edildi; ünlü POI ölçüsü bu takasın ünlüleri götürmediğini söylüyor.

Yan düzeltme: REPORT §5 `RETRIEVED_POIS_PER_DAY = 8` diyor ve "kimsenin bilerek seçmediği knob"
olarak anlatıyordu. `f34de0e` (26 Ağu) değeri 24 yaptı ve gerekçesini ölçümle yazdı; §5 güncellendi.
[#113](https://github.com/yukselburcinn-web/DA592/issues/113) — ilişkili: #63, #79, REPORT §5

### #109. Yemek oturumları da sakin saatini seçsin — 🔒 Kapalı
*(enhancement, `priority:medium`)* #33 saat yarısını kapattı ama sakin-pencere düğümünü yalnızca
gezi POI'lerine verdi. Gerekçe şuydu: yemek boyutu her oturumu zorunlu kılıyor, yani "ne zaman"
modelin seçimi değil. Ölçüldüğünde gerekçenin yarısının yanlış olduğu çıktı — oturum penceresi
hedefinin ±2 saati (`MEAL_WINDOW_HOURS`), yani içeride dört saatlik bir aralık ve seçilecek bir
saat var; bugün o saat kalabalık hiç hesaba katılmadan çözücünün geometrisine bırakılıyor.

Bedeli ölçüldü. #33'ün fazladan yoğunluk ölçüsü türüne ayrıldığında gezi durakları **+10.7'den
−0.4'e** iniyor, yemek durakları ise **+12.1'den ancak +8.2'ye**. Yani #33 sonrası kalan +2.0'nin
neredeyse tamamı yemeklerde. En net vaka Paris / Nightlife Seeker: barlar tam istendiği gibi
kayıyor (Oculto %79'da 22:20'den %23'te 19:24'e) ama akşam yemeği Tour d'Argent'da 20:12'de
**%100** yoğunlukta kalıyor — kendi tipik seviyesi %61.

Çözüm #33'ünkiyle aynı: oturum bandının **içinde** ikinci bir düğüm, restoranın o banttaki en
sakin iki saatine sabitli. Bant genişliğindeki kopya yerinde kalıyor, yani gün ancak akşam
yemeğini en kalabalık saatine sığdırabiliyorsa hâlâ sığdırıyor.

**Yemek ayrı ve daha düşük bir fiyat istiyor** (1500, geziye uygulanan 4000'e karşı) ve sebebi
mekanizmanın kendisi: oturum atlanamadığı için çözücü yüksek fiyata "yemeği bırakarak" cevap
veremiyor — günü sakin saati olan restoranın etrafında yeniden kuruyor, bedelini gezi durakları
ödüyor. Gezi fiyatıyla denendiğinde yemek boşluğu −9.4'e aştı ama gezi boşluğu ters yöne gitti
(−0.4 → +2.1) ve gezi durakların %2.5'ini kaybetti. 1500'de: yemek boşluğu **+7.1 → −5.1**,
gezi −2.1'de, bedel durakların %0.7'si ve mesafede ölçülebilir bir şey yok.

Kontrat her fiyatta korundu: günde 2.0 yemek, 72 günün %100'ünde iki oturum da dolu.

Ölçüm sırasında bir yan bulgu: yüklü makinede üç çözüm 199/603/1035 saniye sürdü, medyan 1.37
saniyeyken. Boş makinede tekrarlandığında `solve_seconds` dışındaki **bütün sütunlar birebir
aynı** çıktı, o üç vaka 1.20/0.97/1.28 saniyeye indi ve 288 çözümün maksimumu 4.60 saniye oldu.
Model deterministik olduğu için aynı girdi aynı işi yapar; fark makinedendi. Harness artık süreyi
ortalama değil medyan raporluyor.
[#109](https://github.com/yukselburcinn-web/DA592/issues/109) — ilişkili: #33, #20, #29, REPORT §5

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

### #32. OSRM demo sunucusundan çık: Aşama 1 self-hosted mesafe, Aşama 2 GTFS transit — ✅ Kapalı
*(enhancement, `priority:medium` — 2026-08-26'da yeniden yazıldı, 2026-08-27'de üç parçası da
bitti: PR #87, #88, #89, #90, #91. Kapsam dışı bırakılan üç iş #93'e taşındı; #93 de aynı gün
kapandı — biri #94'te sürüyor, ikisi aksiyonsuz kapatıldı.)*
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

### #93. #32'den artakalanlar: tek iş günü matrisi, GTFS tazeleme, `use_real_routing` — ✅ Kapalı
*(enhancement, `area:data`/`area:agents`, `priority:medium` — açıldı ve kapandı 2026-08-27)* #32
kapanırken kapsam dışı bırakılan üç şey burada toplanmıştı. Hiçbiri hata değildi; üçü de bilinçli
sadeleştirme ve üçü de REPORT §5'te yazılı. **Kapatma kararı:** 3. madde #94'e taşındı, 1. ve 2.
maddeler için **aksiyon alınmadı** — davranış olduğu gibi duruyor ve belgeli. İkisi de birbirinden ve
#94'ten bağımsız; tekrar gündeme gelirlerse yeni issue açılır.

**1 — Matris tek bir temsili iş gününden üretiliyor — aksiyon alınmadı.** `{PAR,BER}_transit.npz`
feed span'indeki **en dolu Çarşamba**dan geliyor (`build_transit_matrix.py --date` ile değişir),
08:00–20:00 arası 13 kalkışın medyanı alınıyor. Yani "Pazar günü ne olur" sorusunun cevabı yok
(Pazar servisi her iki şehirde de hafta içinin ~%47'si) ve gece tarifesi matriste hiç yok.
**Sebebi yapısal:** TOPTW gezi başına *tek* süre matrisi tüketiyor (`toptw.solve()` `time_m`'i
çözümden önce dolduruyor) ve OR-Tools'ta saate göre değişen ark maliyeti ayrı bir formülasyon
meselesi. Medyan, "tarife o gün sana iyi davrandı mı" şansını dışarıda bırakmak için seçildi.
Yapılmayanlar: hafta sonu için ikinci matris ve `start_date`'in gününe göre seçim; gün başlangıç
saatine göre dilim (Nightlife Seeker 15:00'te başlıyor, sabah zirvesinin medyanı ona yanlış);
TOPTW'de gerçek zaman-bağımlı maliyet.

**2 — GTFS feed'leri hızlı bayatlıyor — aksiyon alınmadı.** IDFM feed'i yalnızca
**2026-08-24 .. 09-25** arasını kapsıyordu, bir aydan az; VBB daha iyi (2026-08-20 .. 12-12) ama o da
sonlu. Sefer saatleri sezonluk değişiyor, OSM yol ağı gibi yıllarca durmuyorlar. Tazeleme **elle**
kaldı: feed'i indir, `build_transit_matrix.py PAR BER --write` koş, commit'le. Yapılmayanlar:
yenileme sıklığına karar (~aylık, feed span'i bitmeden); `build_transit_matrix.py`'ın feed'i kendi
indirmesi (`GTFS_URLS` kayıtlı, indirme adımı yok); uygulamanın dosyadaki `service_date`'e bakıp
bayatlamayı söylemesi; otomatikleşecekse zamanlanmış iş mi yoksa not mu olacağı.

**3 — `use_real_routing` varsayılanı — #94'e taşındı.** Maddeler oraya olduğu gibi eklendi; #94'ün
kabul kriterleri zaten gerçek güzergâh çizimini bayrak açıkken şart koştuğu için varsayılanın
açılıp açılmayacağı o işin parçası.

[#93](https://github.com/yukselburcinn-web/DA592/issues/93) — ilişkili: #32 (kapalı), #94

### #101. README Quickstart'ı eski sekiz şehirli veriyi geri yazıyor; REPORT kendi içinde çelişiyor — ✅ Kapalı
*(bug, documentation, `area:data`, `priority:high` — 2026-08-28)* `fed173e` kataloğu sekiz şehirden
ikiye indirdi; kod bunu takip etti, dokümanların bir kısmı etmedi.

**İlk madde doküman sorunu bile değil: Quickstart takip edilirse uygulama bozuluyor.** README
satır 18-20 üç betiği çalıştırmayı söylüyor (`data/generate_data.py`, `fetch_real_pois.py`,
`fetch_real_demand.py`) ve üçü de **eski sekiz şehirli** veri üretiyor (IST, PAR, ROM, BCN, AMS,
PRG, VIE, LIS). Gönderilen veri iki şehir. `fed173e`'nin kendi commit mesajı sonucu yazıyor: yarı
göç etmiş bir set uygulamayı açmıyor — iki destinasyona karşı sekiz rehber `retrieval/corpus.py`'da
`KeyError`, sekiz talep serisi `forecast_city`'de "less than two full seasonal cycles". README
satır 189 zaten bu betiklerin "kept for history" olduğunu söylüyor, yani **README kendi kendisiyle
çelişiyor**; üstelik satır 24'teki hafifletici cümle yalnızca iki `fetch_*` adımını kapsıyor ve beş
dosyayı birden ezen en yıkıcı olan `generate_data.py` hâlâ zorunlu adım gibi duruyor. Gerçek
üreticiler `pipeline/` altında ve Quickstart onları hiç anmıyor.

**İkinci madde: REPORT kendi içinde tutarsız.** `REPORT.md:47` talep verisini "ülke düzeyi
`tour_occ_nim`, 8 şehir" diye anlatıyor, oysa `pipeline/build_demand.py` NUTS 2 `tour_occ_nin2m`
kullanıyor ve **aynı raporun §5'i doğrusunu yazıyor**. `:228` aynı hatayı tekrarlıyor. `:208`'deki
"ten minutes" çözücü ölçüsü #80'in ölçümüyle değiştirilmeli. BACKLOG'un #33 ve #77 maddeleri de
issue'ların yeniden yazılan başlıklarıyla eşleşmiyor.

Önemi teslimle ilgili: §5'in bütün dürüstlük iddiası her alanın kaynağını tek tek beyan etmesine
dayanıyor (`description_source`, `hours_source`, `price_source`). Aynı raporun iki yerde zıt şey
söylemesi o iddiayı en çok zayıflatan hata türü. (`REPORT.md:200`'ün "eight-city version" ifadesi
bayat değil — geçmişle bilinçli karşılaştırma, kapsam dışı.)

**Kabul kriterleri:**
- [ ] Temiz klondan Quickstart takip edilince uygulama açılıyor, iki şehir de çalışıyor
- [ ] `generate_data.py` / `fetch_real_*` zorunlu adım olarak görünmüyor; ne oldukları tek yerde yazılı
- [ ] REPORT'ta talep granülerliğini anlatan tüm cümleler aynı şeyi söylüyor
- [ ] `REPORT.md:208`'deki çözücü ölçeği tekrar üretilebilir bir ölçüme dayanıyor
- [ ] BACKLOG'un #33 ve #77 maddeleri issue'ların güncel başlık ve kapsamıyla eşleşiyor

**Kapatıldı (2026-08-27), PR #104.** Quickstart'tan veri üretme adımı tamamen kalktı —
uygulamanın okuduğu her dosya commit'li, kurulum dört satır ve sonu `streamlit run app.py`; hiçbir
üretim adımı çalıştırılmadan uygulamanın HTTP 200 verdiği doğrulandı. Üç aşılmış betik yerinde
kaldı ama docstring'lerinde neyi ezdikleri yazılı (`data/legacy/`'ye taşımak ya da silmek ayrı bir
karar olarak açık). REPORT tarafında: `:47` ve `:228`'deki ülke-düzeyi `tour_occ_nim` iddiası NUTS 2
ile hizalandı, çözücü tavanı cümlesi #80'in ölçümüyle değiştirildi, ve issue'nun listelemediği ama
zorunlu hale gelen kısım yapıldı — §3.1 artık `pipeline/build_catalogue.py`'yi ve commit'li dosyanın
sayılarını anlatıyor (654 satır), aşılmış `fetch_real_pois.py`'ye atıf değil.
[#101](https://github.com/yukselburcinn-web/DA592/issues/101) — ilişkili: `fed173e`, #80 (kapalı), #33, #77, #71 (kapalı)

---

## Öncelik sırası (2026-08-28 itibarıyla açık işler)

| Öncelik | Issue | Neden |
|---|---|---|
| 1 | #7 | `priority:high` etiketli, final rapor için gerçek LLM sonucu eksik |
| 2 | #92 | #32 Aşama 2'nin açıkta bıraktığı tek ölçülebilir yanlışlık: hub koordinatı yüzünden BER erişimi beş kat sapıyordu ve şu an düzeltilmiyor, telafi ediliyor. Veri işi, router'a dokunmuyor |
| 3 | #94 | Harita, planın söylediği mesafenin yarısını çiziyor. #32'den sonra çelişki büyüdü; sokak modları için gereken veri zaten depoda |
