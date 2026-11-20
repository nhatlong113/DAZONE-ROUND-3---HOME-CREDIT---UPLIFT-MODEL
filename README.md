# DAZONE 2026 Round 3.1 - Home Credit Uplift Modeling

Repo này là bài làm của nhóm cho vòng 3.1 cuộc thi DAZONE 2026, đề bài do Home Credit đưa ra. Câu hỏi chính khá đơn giản: với mỗi khách hàng, nên chào Cash Loan hay Credit Card trước thì toàn danh mục sẽ chuyển đổi tốt hơn?

Sau khá nhiều lần thử, nhóm chốt mô hình cuối là ensemble của 3 cấu hình UPLIFT_RF_CIT và 2 cấu hình INT_RF. Trên 5-fold CV, Qini_total trung bình đạt 0.094930, fold tốt nhất đạt 0.132108.

## 1. Bài toán

| Thành phần | Định nghĩa |
|---|---|
| Treatment | Chiến lược truyền thông được phân bổ ngẫu nhiên: `cash` hoặc `card` |
| Outcome | `cash_sign_30d` (ký Cash Loan), `card_sign_30d` (mở Credit Card), không hủy, trong 30 ngày sau treatment |
| Uplift Cash | `P(cash_sign_30d = 1 \| Cash, x) - P(cash_sign_30d = 1 \| Card, x)` |
| Uplift Card | `P(card_sign_30d = 1 \| Card, x) - P(card_sign_30d = 1 \| Cash, x)` |
| Metric | `Qini_total = Qini_cash + Qini_card` |

### Qini Coefficient

Nhóm xếp khách hàng theo uplift dự đoán từ cao xuống thấp, rồi tại mỗi tỷ lệ dân số $\phi = k/N$ tính:

$$Q(\phi) = \frac{\sum_{i \le k} y_i\, w_i}{N_{treatment}} - \frac{\sum_{i \le k} y_i\,(1-w_i)}{N_{control}}$$

$$AUC_{random} = \frac{Q(1)}{2}, \qquad \text{Qini}_{coef} = \frac{AUC_{model} - AUC_{random}}{AUC_{perfect} - AUC_{random}}$$

AUC được tính bằng tích phân hình thang trên đường `(φ, Q(φ))`. `AUC_perfect` lấy từ trường hợp lý tưởng, khi điểm số chính là `w·y - (1-w)·y`. Với target cash thì `w = 1` khi khách nhận treatment Cash, còn với target card thì `w = 1` khi khách nhận treatment Card. Đề bài chỉ định nghĩa Qini là `AUC_model - AUC_random`, nhóm chia thêm cho phần của mô hình lý tưởng để điểm của cash và card cùng thang đo trước khi cộng lại.

## 2. Dữ liệu

| Tập | Số dòng | Số cột | Ghi chú |
|---|---:|---:|---|
| `data/raw/dazone_sample.csv` | 136,992 | 44 | `customer_id`, `treatment`, 2 target, 40 feature |
| `data/raw/dazone_sample_test.csv` | 34,080 | 44 | Đã ẩn nhãn |

| Treatment | Khách hàng | Tỷ trọng |
|---|---:|---:|
| Cash | 75,612 | 55.19% |
| Card | 61,380 | 44.81% |

| Target | Số dương | Tỷ lệ |
|---|---:|---:|
| `cash_sign_30d` | 8,108 | 5.92% |
| `card_sign_30d` | 3,714 | 2.71% |
| Không chuyển đổi | | 91.82% |

Tỷ lệ chuyển đổi theo treatment (%):

| | Cash Sign | Card Sign |
|---|---:|---:|
| Cash Treatment | 6.26 | 2.61 |
| Card Treatment | 5.50 | 2.84 |
| Chênh lệch (ATE) | +0.76 pp | +0.24 pp |

Có thể thấy chiến lược Cash chỉ làm tỷ lệ ký Cash tăng thêm khoảng 0.76 điểm %, còn chiến lược Card làm tỷ lệ mở thẻ tăng khoảng 0.24 điểm %. Mức chênh này nhỏ so với tỷ lệ nền, nên cái nhóm cần không phải là đoán ai sẽ chuyển đổi, mà là tìm ra ai thật sự bị tác động bởi từng kiểu truyền thông.

## 3. Pipeline

```mermaid
flowchart LR
    A[Raw data] --> B[01 Preprocessing]
    B --> C[02 Visualization]
    B --> D[03 Feature Interaction]
    B --> E[04 Random Search]
    E --> F[05 Ensemble CV]
    F --> G[06 Inference]
    G --> H[07 Feature Importance]
```

| Bước | Notebook | Đầu ra |
|---|---|---|
| Data Preprocessing | [01_data_preprocessing](notebooks/01_data_preprocessing.ipynb) | `df_feature.csv`, `df_target.csv`, `test_df_feature.csv`, `transformers.joblib` |
| Visualization | [02_visualization_business_insight](notebooks/02_visualization_business_insight.ipynb) | Biểu đồ EDA theo phân khúc |
| Feature Interaction | [03_feature_interaction](notebooks/03_feature_interaction.ipynb) | Correlation, uplift theo phân khúc |
| Random Search | [04_uplift_random_search](notebooks/04_uplift_random_search.ipynb) | `results/random_search_*.csv` |
| Ensemble CV | [05_ensemble_cross_validation](notebooks/05_ensemble_cross_validation.ipynb) | `results/ensemble_cv_qini.csv`, Qini curve |
| Inference | [06_inference_test](notebooks/06_inference_test.ipynb) | Xác suất, uplift, chiến lược khuyến nghị |
| Feature Importance | [07_feature_importance](notebooks/07_feature_importance.ipynb) | `feature_importance.csv` |

## 4. Data Preprocessing

Đầu tiên nhóm bỏ `customer_id` và 6 cột bị thiếu hơn một nửa dữ liệu:

| Cột | Missing |
|---|---:|
| `cnt_month_from_first_transaction` | 93.60% |
| `cnt_month_from_last_transaction` | 93.60% |
| `ratio_vol_hcvn_cash_appl_12m` | 89.47% |
| `ratio_monthly_amt_l6m_vs_n6m` | 82.55% |
| `max_cntr_len_fi_12m` | 74.07% |
| `f_appointment_3m` | 55.70% |

Các cột còn lại được chia theo kiểu dữ liệu để xử lý. Mọi bước đều fit trên tập train, sau đó dùng lại y nguyên cho tập test:

| Nhóm | Số cột | Xử lý |
|---|---:|---|
| Số liên tục | 15 | `SimpleImputer(median)`, `MinMaxScaler` |
| Số rời rạc, flag | 12 | `SimpleImputer(most_frequent)`, `MinMaxScaler` |
| Nominal | 7 | `SimpleImputer(most_frequent)`, `OneHotEncoder(drop="first")`, ra 35 cột (gồm `treatment_cash`) |
| Ordinal `cli_education` | 1 | `SimpleImputer(most_frequent)`, `OrdinalEncoder` theo thứ tự ELEMENTARY_SCHOOL, PTCS, PTTH, BACHELORS, MASTERS, `MinMaxScaler` |

Sau bước này, `df_feature` có 136,992 dòng và 63 cột (62 feature đã đưa về khoảng [0, 1] cùng cột `treatment_cash`), `df_target` gồm hai cột `cash_sign_30d` và `card_sign_30d`.

## 5. Visualization và Business Insight

Phần này nhóm so sánh tỷ lệ ký Cash của khách nhận chiến lược Cash với tỷ lệ mở thẻ của khách nhận chiến lược Card, trong từng nhóm khách hàng. Cột `cash_minus_card` là hiệu của hai tỷ lệ đó, tính theo điểm %.

<img src="results/figures/eda_treatment_cross_conversion.png" width="85%">

| Phân khúc | Nhận xét | Số liệu |
|---|---|---|
| Risk group | Rủi ro càng cao thì chuyển đổi càng thấp ở cả hai sản phẩm | A: Cash 13.76%, Card 6.70%; E: Cash 2.20%, Card 1.05% |
| Số lần thanh toán | Chuyển đổi tăng gần như đều theo decile | Cash 1.51% (D1) lên 18.11% (D10); Card 0.99% lên 5.93% |
| Tất toán khoản vay 12 tháng | Khách tất toán nhiều khoản phản ứng mạnh với Cash | Cash 7.46% (0 khoản), 26.35% (4 khoản trở lên, n = 260) |
| Hạn mức offer | Decile cao nhất có chuyển đổi cao nhất | D10: Cash 13.04%, Card 6.30% |
| Học vấn | Card cạnh tranh tốt hơn ở nhóm học vấn cao | `cash_minus_card` PTTH 4.03, MASTERS 1.98; Card ở MASTERS 4.78% |
| Tuổi | Nhóm dưới 25 tuổi mở Card nhiều nhất | Card 4.47%; nhóm 55+ có chênh lệch lớn nhất 6.60 |
| Worst credit status | Status 3 là nhóm duy nhất Card cao hơn Cash | Cash 1.01%, Card 1.39% (n = 171) |
| Đặt lịch hẹn 3 tháng | Cash rất cao nhưng mẫu quá nhỏ | Cash 48.28%, Card 4.00% (54 khách) |
| Telesales kết nối | Tăng từ D1 đến D5 rồi đi ngang | Cash 4.99% (D1), 6.93% (D5), sau đó 6.03% đến 6.61% |

<img src="results/figures/eda_risk_group.png" width="49%"> <img src="results/figures/eda_payment_decile.png" width="49%">

Lưu ý là các nhóm rủi ro F, G, H, I chỉ có lần lượt 69, 13, 5 và 4 khách, nên nhóm không rút ra kết luận gì từ các nhóm này.

## 6. Feature Interaction

### 6.1 Tương quan giữa các feature (|corr| ≥ 0.7)

| Feature 1 | Feature 2 | corr |
|---|---|---:|
| `cnt_month_from_first_transaction` | `cnt_month_from_last_transaction` | 0.84 |
| `min_sign_vol` | `avg_sign_vol` | 0.75 |
| `cnt_month_from_last_appl` | `cnt_month_from_first_transaction` | 0.73 |
| `cnt_month_from_last_appl` | `cnt_month_from_last_transaction` | 0.71 |

### 6.2 Tương tác giữa feature và treatment

Do khách được chia ngẫu nhiên vào hai chiến lược, nhóm có thể ước lượng uplift của từng phân khúc bằng cách lấy tỷ lệ của nhóm nhận chiến lược trừ đi tỷ lệ của nhóm còn lại. Để xem feature nào làm uplift thay đổi nhiều, nhóm chia mỗi feature thành các bin rồi tính độ lệch chuẩn có trọng số của uplift giữa các bin. Những bin có dưới 200 khách ở một trong hai nhóm bị loại để tránh nhiễu.

<img src="results/figures/interaction_feature_treatment.png" width="70%">

| Phân khúc | `uplift_cash` (pp) | `uplift_card` (pp) |
|---|---:|---:|
| `cnt_month_from_last_appl` ≤ 6 tháng | +4.43 | +0.74 |
| `cnt_month_from_last_appl` 6 đến 19 tháng | +2.03 / +2.03 | +0.33 / -0.02 |
| `cnt_month_from_last_appl` > 28 tháng | -0.52 đến +0.18 | -0.09 đến +0.49 |
| `risk_grp` A / B | +2.42 / +2.30 | +1.07 / +0.66 |
| `risk_grp` E | -0.07 | -0.04 |
| `cnt_total_payment` D9 / D10 | +2.79 / +3.39 | +0.35 / +0.85 |
| `cnt_total_payment` D1 đến D6 | -0.22 đến +0.38 | -0.12 đến +0.28 |

Nhìn chung, chiến lược Cash có tác dụng rõ nhất với những khách vừa nộp hồ sơ gần đây, thuộc nhóm rủi ro thấp (A, B) và có nhiều lần thanh toán. Ngược lại, với khách rủi ro E hoặc đã lâu không nộp hồ sơ, chào Cash gần như không làm tăng thêm chuyển đổi. Kết quả này khớp với phần feature importance ở mục 8, nơi `cnt_month_from_last_appl` cũng đứng đầu.

<img src="results/figures/interaction_risk_age.png" width="95%">

### 6.3 Ma trận đầu vào của INT_RF

$$X_{aug} = [\,X,\; W,\; X \cdot W\,] \in \mathbb{R}^{n \times 125}, \qquad \text{uplift}(x) = \hat P(y=1 \mid x, W=1) - \hat P(y=1 \mid x, W=0)$$

## 7. Uplift Modelling

### 7.1 Sàng lọc model

| Meta-learner | Base model | Direct uplift (tree) | Direct uplift (forest) |
|---|---|---|---|
| T, S, INT, CT, TO, X, R, DR | dt, rf, extratrees, adaboost, lgbm, xgb, cat, hgb, logistic, knn, fuzzy_knn, svm, mlp, kan, deep_kan | UPLIFT_TREE_{KL, ED, CHI, CTS, DDP}, INTERACTION_TREE, CAUSAL_INFERENCE_TREE | UPLIFT_RF_{KL, ED, CHI, CTS, DDP, IT, CIT}, UTBOOST |

Nhóm chạy thử rất nhiều tổ hợp meta-learner, base model và các mô hình uplift trực tiếp, mỗi tổ hợp đánh giá bằng K-fold. Dưới đây là 10 tổ hợp tốt nhất (trích từ Bảng 6 trong `docs/report.pdf`):

| # | Model | Qini cash | Qini card | Qini total |
|---:|---|---:|---:|---:|
| 1 | INT_RF | +0.05424 | +0.03991 | +0.09415 |
| 2 | UPLIFT_RF_CIT | +0.05721 | +0.03251 | +0.08972 |
| 3 | INT_EXTRATREES | +0.04963 | +0.02785 | +0.07748 |
| 4 | TO_KAN | +0.05628 | +0.01970 | +0.07598 |
| 5 | UPLIFT_TREE_CIT | +0.05575 | +0.01855 | +0.07430 |
| 6 | DR_SVM | +0.04458 | +0.02937 | +0.07395 |
| 7 | TO_RF | +0.05829 | +0.01354 | +0.07183 |
| 8 | UPLIFT_RF_IT | +0.05722 | +0.01402 | +0.07124 |
| 9 | TO_LOGISTIC | +0.04885 | +0.02201 | +0.07086 |
| 10 | UPLIFT_RF_KL | +0.05637 | +0.01311 | +0.06948 |

Hai mô hình đứng đầu là INT_RF (S-Learner có thêm tương tác feature với treatment, dùng Random Forest) và UPLIFT_RF_CIT (Uplift Random Forest của CausalML với tiêu chí CIT). Nhóm tập trung tối ưu tiếp hai mô hình này.

### 7.2 Random Search

Mỗi cấu hình được đánh giá bằng Qini trên dự đoán out-of-fold của 5-fold, cash và card dùng bộ fold riêng. Các cấu hình có dấu `*` là những cấu hình nhóm đưa vào ensemble.

UPLIFT_RF_CIT, 30 cấu hình:

| Cấu hình | n | depth | leaf | treat | mf_cap | Qini total |
|---|---:|---:|---:|---:|---:|---:|
| CFG028 * | 80 | 8 | 120 | 20 | 32 | 0.107228 |
| CFG019 | 60 | 8 | 220 | 30 | 16 | 0.101231 |
| CFG009 * | 120 | 5 | 80 | 30 | 32 | 0.099346 |
| CFG011 * | 120 | 5 | 160 | 40 | 32 | 0.099099 |
| CFG014 | 220 | 5 | 160 | 30 | 20 | 0.094458 |

INT_RF, 40 cấu hình:

| Cấu hình | n | depth | leaf | split | max_features | criterion | class_weight | Qini total |
|---|---:|---:|---:|---:|---|---|---|---:|
| RS_040 | 180 | 6 | 160 | 20 | 0.75 | entropy | None | 0.098835 |
| RS_001 | 320 | 9 | 80 | 2 | sqrt | gini | balanced_subsample | 0.094145 |
| RS_034 | 420 | 8 | 120 | 20 | log2 | gini | None | 0.091325 |
| RS_027 * | 560 | 6 | 80 | 50 | log2 | entropy | balanced | 0.091053 |
| RS_028 * | 420 | 6 | 30 | 2 | log2 | gini | balanced | 0.090592 |

Kết quả đầy đủ của từng cấu hình và từng fold nằm trong `results/random_search_uplift_rf_cit.csv`, `results/random_search_int_rf.csv` và các file `*_folds.csv` đi kèm. Với UPLIFT_RF_CIT, nhóm bỏ qua CFG019 dù điểm cao thứ hai vì `qini_card` ở fold 5 bị âm (-0.00107), trong khi ba cấu hình được chọn đều dương ở cả 5 fold. Riêng CFG012 (`max_depth=None`) bị lỗi lúc train nên không có điểm.

### 7.3 Model cuối: Ensemble 3 UPLIFT_RF_CIT + 2 INT_RF

$$\text{uplift}_{ensemble}(x) = \frac{1}{5}\sum_{m=1}^{5}\text{uplift}_m(x)$$

Thay vì chọn một mô hình duy nhất, nhóm lấy trung bình uplift của 5 mô hình (làm riêng cho cash và card) để kết quả ổn định hơn. Ensemble được đánh giá bằng `StratifiedKFold(5, shuffle=True, random_state=42)`, chia theo tổ hợp `treatment_cash`, `cash_sign_30d`, `card_sign_30d`:

| Fold | Qini cash | Qini card | Qini total |
|---|---:|---:|---:|
| 0 | 0.054649 | 0.034239 | 0.088888 |
| 1 | 0.060871 | 0.071237 | 0.132108 |
| 2 | 0.053167 | 0.036965 | 0.090131 |
| 3 | 0.053593 | 0.023256 | 0.076850 |
| 4 | 0.065876 | 0.020799 | 0.086675 |
| Trung bình | 0.057631 | 0.037299 | 0.094930 |

Qini curve của fold 1:

<img src="results/figures/qini_curve_cash_ensemble_best_fold.png" width="49%"> <img src="results/figures/qini_curve_card_ensemble_best_fold.png" width="49%">

Bảng dưới là Qini tính trên chính tập train sau khi fit lại trên toàn bộ dữ liệu. Vì là điểm in-sample nên số khá cao, nhóm chỉ để tham khảo chứ không dùng để so sánh mô hình:

| Model | Qini cash | Qini card | Qini total |
|---|---:|---:|---:|
| CFG028_UPLIFT_RF_CIT | 0.134977 | 0.232503 | 0.367480 |
| CFG009_UPLIFT_RF_CIT | 0.068274 | 0.069925 | 0.138199 |
| CFG011_UPLIFT_RF_CIT | 0.068305 | 0.069816 | 0.138122 |
| INT_RF_RS_027 | 0.088286 | 0.144436 | 0.232722 |
| INT_RF_RS_028 | 0.091589 | 0.167941 | 0.259530 |
| Ensemble | 0.088185 | 0.158916 | 0.247102 |

### 7.4 Inference

Khi dự đoán cho tập test, nhóm fit lại cả 5 mô hình trên toàn bộ 136,992 dòng train. Xác suất của UPLIFT_RF_CIT được tính bằng tỷ lệ nền của nhóm control cộng với uplift (cắt về khoảng [0, 1]), còn INT_RF cho ra xác suất trực tiếp khi gán khách vào chiến lược tương ứng. Khách nào có `uplift_cash_score - uplift_card_score ≥ 0` thì nên chào Cash trước, còn lại chào Card. File nhóm đã nộp là `results/final_submission/submission.csv`.

## 8. Feature Importance

$$FI^{(k)}_j = \frac{1}{5}\sum_{m=1}^{5} FI^{(k)}_{j,m}, \quad k \in \{lime, shap, permutation\} \qquad FI^{final}_j = \frac{FI^{(lime)}_j + FI^{(shap)}_j + FI^{(permutation)}_j}{3}$$

Nhóm dùng ba cách đo cho từng mô hình: LIME và SHAP (KernelExplainer) đo mức đóng góp của feature vào uplift score, còn permutation đo xem Qini_cash + Qini_card giảm bao nhiêu khi xáo trộn feature đó (trên 800 dòng train, lặp 3 lần). Mỗi cách được lấy trung bình qua 5 mô hình, sau đó lấy trung bình ba cách để ra điểm cuối.

<img src="results/figures/feature_importance_top15.png" width="75%">

| # | Feature | LIME | SHAP | Permutation | FI final |
|---:|---|---:|---:|---:|---:|
| 1 | `cnt_month_from_last_appl` | 0.009252 | 0.007944 | 0.113435 | 0.043544 |
| 2 | `cnt_report_institution_12m` | 0.006752 | 0.002000 | 0.090472 | 0.033074 |
| 3 | `cli_emp_period` | 0.001958 | 0.001191 | 0.034610 | 0.012586 |
| 4 | `cli_age` | 0.001285 | 0.000690 | 0.034755 | 0.012243 |
| 5 | `last_offer_limit` | 0.002552 | 0.001004 | 0.032710 | 0.012089 |
| 6 | `last_process_channel_D` | 0.000000 | 0.000270 | 0.031456 | 0.010575 |
| 7 | `cnt_fi_inst_appl_12m` | 0.007850 | 0.001796 | 0.007634 | 0.005760 |
| 8 | `risk_grp_E` | 0.003746 | 0.001023 | 0.012019 | 0.005596 |
| 9 | `cli_marital_status_Married` | 0.000000 | 0.000108 | 0.015802 | 0.005303 |
| 10 | `risk_grp_C` | 0.000584 | 0.000161 | 0.012952 | 0.004566 |

Cần lưu ý là permutation tính theo Qini nên có giá trị lớn hơn hẳn LIME và SHAP, vì vậy thứ hạng cuối cùng chủ yếu do permutation quyết định. Bảng đầy đủ cho 62 feature và cho từng mô hình nằm trong `results/final_submission/feature_importance/`.

## 9. Thử nghiệm khác

Trong quá trình làm, nhóm có đọc và thử áp dụng một số paper lên INT_RF và UPLIFT_RF_CIT (dùng cấu hình mặc định ban đầu). Kết quả không tốt hơn rõ ràng nên không được đưa vào mô hình cuối, nhưng nhóm vẫn để lại đây để tham khảo:

| Paper | Model | Qini cash | Qini card | Qini total |
|---|---|---:|---:|---:|
| Baseline | INT_RF (RS_001) | 0.054236 | 0.039909 | 0.094145 |
| Baseline | UPLIFT_RF_CIT (CFG001) | 0.057206 | 0.032514 | 0.089720 |
| Feature Selection for Uplift (Zhao et al.), Chi filter | CHI_FS_INT_RF | 0.058476 | 0.039081 | 0.097557 |
| CAFE | CAFE_UPLIFT_RF_CIT | 0.057509 | 0.034297 | 0.091806 |
| CAFE | CAFE_INT_RF | 0.055961 | 0.032946 | 0.088907 |
| EFIN | EFIN_UPLIFT_RF_CIT | 0.055012 | 0.029427 | 0.084439 |
| EFIN | EFIN_INT_RF | 0.024991 | 0.019882 | 0.044873 |
| Class Flipping | CLASS_FLIP_INT_RF | 0.051609 | 0.012721 | 0.064329 |
| Class Flipping | CLASS_FLIP_UPLIFT_RF_CIT | 0.042424 | 0.013428 | 0.055852 |
| RUAD | RUAD_SNN | 0.034764 | 0.016639 | 0.051403 |

Notebook Chi filter mới chạy xong phần INT_RF, phần UPLIFT_RF_CIT dừng ở fold 2 nên chưa có điểm. Ngoài ra nhóm còn thử Spline KAN, Deep KAN, TabNet và mạng nhiều đầu ra theo hướng PTONet, code nằm trong `experiments/model_variations/` nhưng các notebook này không lưu lại output.

## 10. Cấu trúc thư mục

```
.
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── analysis.py
│   ├── metrics.py
│   ├── models.py
│   ├── search.py
│   ├── ensemble.py
│   ├── explain.py
│   └── plots.py
├── notebooks/            01 đến 07
├── data/
│   ├── raw/
│   └── processed/
├── results/
│   ├── final_submission/
│   ├── figures/
│   └── *.csv
├── experiments/
├── archive/              notebook gốc chạy trên Kaggle, file zip nộp bài
└── docs/                 đề bài, mô tả dữ liệu, report.pdf
```

## 11. Cách chạy

```bash
pip install -r requirements.txt
```

```bash
jupyter notebook notebooks/
```

Các notebook chạy theo thứ tự từ 01 đến 07. Notebook 04 khá nặng: trên Kaggle (CPU 4 nhân) nhóm mất khoảng 650 phút cho UPLIFT_RF_CIT và 576 phút cho INT_RF, các notebook 05 đến 07 cũng mất khá lâu. Nếu chỉ muốn xem kết quả thì có thể mở thẳng thư mục `results/`, số liệu ở đó lấy từ các lần chạy trên Kaggle (notebook gốc để trong `archive/original_notebooks/`). Nếu chạy lại với phiên bản `scikit-learn` hoặc `causalml` khác, điểm có thể lệch nhẹ ở chữ số thập phân thứ ba.
