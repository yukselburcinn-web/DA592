# RoamWise Backlog

Bu liste `REPORT.md` §5 (Known limitations) ve §7 (Suggested next steps) temel alınarak
hazırlandı — her madde GitHub'da doğrudan yeni bir Issue olarak açılabilecek şekilde
formatlanmıştır (başlık, öneri edilen etiketler, açıklama, kabul kriterleri).

Alan sahipliği önerisi (isteğe göre değiştirin):
- **Alan A — Veri & Modeller**: `data/`, `models/`
- **Alan B — Retrieval & Bilgi Grafiği**: `retrieval/`, `knowledge_graph/`
- **Alan C — Ajanlar, Orkestrasyon & UI**: `agents/`, `app.py`, `evaluation/`

## Nasıl kullanılır

1. Her başlığı GitHub'da **New Issue** ile açın, açıklama kısmına ilgili gövdeyi yapıştırın.
2. Etiketleri GitHub'da yoksa önce oluşturun (Settings → Labels): `area:data`, `area:retrieval`,
   `area:agents`, `area:ui`, `priority:high`, `priority:medium`, `priority:low`, `good-first-issue`.
3. Bir GitHub Project (Kanban board) oluşturup issue'ları `Backlog / In Progress / Review / Done`
   kolonlarına ekleyin.

---

## Alan A — Veri & Modeller

### 1. Sentetik veriyi gerçek kaynaklarla değiştir
**Etiketler:** `area:data`, `priority:high`
**Açıklama:** `data/generate_data.py` şu an prosedürel sentetik veri üretiyor (proposal'da
Kaggle/TripAdvisor/OpenStreetMap/Wikidata belirtilmişti ama sandbox'ta kimlik doğrulamalı
erişim yoktu). Gerçek OpenStreetMap (Overpass API) POI verisi ve Wikidata SPARQL sorgularıyla
şehir/POI verisini, gerçek bir turizm/havacılık talep veri setiyle (UNWTO, Eurostat vb.)
`demand_timeseries.csv`'yi değiştirin.
**Kabul kriterleri:**
- [ ] En az 3 şehir gerçek OSM POI verisiyle geliyor
- [ ] `poi.csv` şeması korunuyor (mevcut modüller değişmeden çalışıyor)
- [ ] Gerçek talep verisi kaynağı README'de belgelendi

### 2. POI kataloğunu şehir başına büyüt
**Etiketler:** `area:data`, `priority:medium`
**Açıklama:** Şu an şehir başına 10 POI var — bu, karşılaştırmalı analizde Fusion RAG'in
Hybrid RAG'e karşı multi-hop recall avantajının net görünmesini engelliyor (bkz. REPORT.md §3.5).
Şehir başına 50-100+ POI'ye çıkarıp `evaluation/comparative_analysis.py`'yi tekrar çalıştırın.
**Kabul kriterleri:**
- [ ] Şehir başına en az 50 POI
- [ ] Karşılaştırmalı analiz tekrar çalıştırıldı, sonuçlar REPORT.md'de güncellendi

### 3. Forecasting modelini genişlet / karşılaştır
**Etiketler:** `area:data`, `priority:low`, `good-first-issue`
**Açıklama:** `models/forecasting.py` Holt-Winters kullanıyor (Prophet/LSTM yerine, bkz.
REPORT.md §3.2). Prophet ile bir karşılaştırma ekleyip hangi modelin bu veri hacminde daha
iyi performans gösterdiğini raporlayın.
**Kabul kriterleri:**
- [ ] Prophet tabanlı alternatif `models/forecasting_prophet.py` olarak eklendi
- [ ] İki modelin MAE/RMSE karşılaştırması bir tabloda sunuldu

---

## Alan B — Retrieval & Bilgi Grafiği

### 4. Semantic search'ü gerçek embedding modeline geçir
**Etiketler:** `area:retrieval`, `priority:high`
**Açıklama:** `retrieval/semantic_search.py` şu an TF-IDF+LSA kullanıyor (indirme gerektiren
transformer modeli yerine, bkz. REPORT.md §3.3). `sentence-transformers` (örn. `all-MiniLM-L6-v2`)
+ FAISS ile değiştirin; `SemanticIndex.encode` arayüzünü koruyun ki diğer modüller etkilenmesin.
**Kabul kriterleri:**
- [ ] `SemanticIndex` aynı public arayüzle çalışıyor
- [ ] `tests/test_pipeline.py` hâlâ geçiyor
- [ ] requirements.txt güncellendi

### 5. Neo4j'ye opsiyonel geçiş
**Etiketler:** `area:retrieval`, `priority:low`
**Açıklama:** `knowledge_graph/build_graph.py` NetworkX kullanıyor. `GraphIndex` sınıfının
metodlarını (bkz. REPORT.md §3.3) Neo4j Python driver'ı ile yeniden implemente eden alternatif
bir backend ekleyin (bir config flag ile seçilebilir olsun).
**Kabul kriterleri:**
- [ ] `GraphIndex` aynı public metodlarla (city_pois, multi_hop_transport_to_poi, vb.) çalışıyor
- [ ] README'de Neo4j kurulum talimatı var

### 6. Multi-hop sorgu seti genişlet
**Etiketler:** `area:retrieval`, `priority:medium`
**Açıklama:** `evaluation/comparative_analysis.py`'deki `TEST_QUERIES` sadece 8 sorgu içeriyor
ve literal anahtar kelime örtüşmesine bağımlı. Query metninde "transport/near" gibi kelimeler
geçmeyen, sadece grafik çıkarımıyla çözülebilecek yeni sorgular ekleyin (örn. "sabah varışından
sonra ilk gün için uygun yerler").
**Kabul kriterleri:**
- [ ] En az 10 yeni multi-hop sorgu eklendi
- [ ] Fusion RAG'in Hybrid RAG'e karşı recall farkı ölçülebilir şekilde arttı

---

## Alan C — Ajanlar, Orkestrasyon & UI

### 7. Gerçek LLM entegrasyonunu uçtan uca test et
**Etiketler:** `area:agents`, `priority:high`
**Açıklama:** `agents/llm_client.py::AnthropicLLMClient` kodda var ama hiç canlı API anahtarıyla
test edilmedi. `ANTHROPIC_API_KEY` ile tüm ajan akışını (Forecaster, FusionRAG, Router, orchestrator)
çalıştırıp çıktı kalitesini değerlendirin; `evaluation/comparative_analysis.py::run_llm_hallucination_probe`'u
gerçek sonuçlarla REPORT.md'ye ekleyin.
**Kabul kriterleri:**
- [ ] Canlı LLM ile üretilen örnek bir itinerary REPORT.md'ye eklendi
- [ ] Hallucination probe sonuçları raporlandı

### 8. Rota optimizasyonunu gerçek yol ağıyla geliştir
**Etiketler:** `area:agents`, `priority:medium`
**Açıklama:** `optimization/routing.py` düz çizgi (haversine) mesafe ve sabit 4.5km/h yürüme
hızı varsayıyor. OSRM/OpenRouteService gibi bir yönlendirme API'siyle gerçek yürüme/toplu taşıma
sürelerini kullanacak şekilde güncelleyin. Açılış saatlerini de modele ekleyin.
**Kabul kriterleri:**
- [ ] Gerçek yol mesafesi/süresi kullanılıyor
- [ ] POI açılış saatleri itinerary'de dikkate alınıyor

### 9. LangGraph'a opsiyonel geçiş
**Etiketler:** `area:agents`, `priority:low`
**Açıklama:** `agents/orchestrator.py` özel bir state-machine kullanıyor (bkz. REPORT.md §3.4).
Aynı akışı LangGraph `StateGraph` ile yeniden implemente eden alternatif bir orchestrator ekleyin,
performans/okunabilirlik açısından karşılaştırın.
**Kabul kriterleri:**
- [ ] `agents/orchestrator_langgraph.py` aynı `plan_trip()` arayüzüyle çalışıyor
- [ ] İki yaklaşımın artı/eksileri REPORT.md'de karşılaştırıldı

### 10. Streamlit UI iyileştirmeleri
**Etiketler:** `area:ui`, `priority:medium`, `good-first-issue`
**Açıklama:** Mobil görünümde sidebar varsayılan kapalı açılıyor, harita bazı durumlarda geç
render oluyor. Responsive davranışı iyileştirin; fiyat/süre filtreleri, çoklu şehir (multi-city
trip) desteği gibi UX geliştirmeleri ekleyin.
**Kabul kriterleri:**
- [ ] Mobil genişlikte sidebar/harita düzgün çalışıyor
- [ ] En az 1 yeni filtre/özellik eklendi

### 11. CI pipeline kur
**Etiketler:** `area:infra`, `priority:medium`, `good-first-issue`
**Açıklama:** `.github/workflows/tests.yml` ekleyin — her PR'da `pytest roamwise/tests/ -v`
otomatik çalışsın. Böylece 3 kişi paralel çalışırken kırılan bir modül fark edilmeden main'e
girmez.
**Kabul kriterleri:**
- [ ] PR açıldığında GitHub Actions testleri otomatik çalıştırıyor
- [ ] README'ye CI badge eklendi

### 12. Dockerfile / deploy hazırlığı
**Etiketler:** `area:infra`, `priority:low`
**Açıklama:** Uygulamayı Streamlit Community Cloud veya bir Docker container'ında paylaşılabilir
hale getirin, böylece demo için herkesin yerel kurulum yapması gerekmez.
**Kabul kriterleri:**
- [ ] `Dockerfile` eklendi, `docker build && docker run` ile çalışıyor
- [ ] (Opsiyonel) Canlı demo linki README'ye eklendi

---

## Öneri: sprint / hafta planı

| Hafta | Odak |
|---|---|
| 1 | Alan sahipliğini netleştirin, Issue #11 (CI) ilk tamamlanacak iş olsun ki sonraki PR'lar güvenli olsun |
| 2-3 | Her alan kendi yüksek öncelikli issue'sunu (#1, #4, #7) bitirir |
| 4 | Orta öncelikli issue'lar (#2, #6, #8, #10) + haftalık kısa senkron |
| 5 | Düşük öncelikli / opsiyonel issue'lar (#3, #5, #9, #12) + REPORT.md güncellemesi |
