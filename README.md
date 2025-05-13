
## 🎧 **HitSense**

### *"Predict the streams, before it dreams."*

---

### 🧠 **Mission Objective**

Build a machine learning model that predicts the **number of streams** a song might get, using its metadata like danceability, energy, tempo, platform presence, etc.

---

### 📦 **Dataset**

This dataset includes detailed metadata for the most popular songs of 2023. It contains not just musical features, but also data on streaming platforms like Spotify, Apple Music, Deezer, and Shazam.

**Key Features:**

* `track_name`, `artist(s)_name`
* `released_year`, `released_month`, `released_day`
* Streaming presence: Spotify, Apple, Deezer, Shazam
* Musical features:
  `bpm`, `key`, `mode`, `danceability_%`, `valence_%`, `energy_%`,
  `acousticness_%`, `liveness_%`, `speechiness_%`

---

### 🔍 **Workflow**

| Step | Task                                         | Status |
| ---- | -------------------------------------------- | ------ |
| 1    | Load & understand dataset                    | ✅      |
| 2    | Clean missing / irrelevant columns           | ✅      |
| 3    | Exploratory Data Analysis (EDA)              | 🔄     |
| 4    | Feature Engineering & Transformation         | ✅      |
| 5    | Encoding categorical variables               | ✅      |
| 6    | Visual comparison before vs after transforms | ⏳      |
| 7    | Model Training                               | ⏳      |
| 8    | Evaluation                                   | ⏳      |
| 9    | (Optional) Deploy with Streamlit             | ⏳      |

---

### 🧹 **Current Cleaned Dataset**

* `new_df`: Cleaned and transformed dataset
* `streams`: Target variable (regression)
* All highly skewed columns were Yeo-Johnson transformed
* Categorical columns encoded:

  * `key`: Ordinal encoding
  * `mode`: Binary encoding

---

### 🧪 **Next Up**

* EDA visualizations comparing **original vs transformed** columns
* Start training regression models (e.g., **Random Forest**, **XGBoost**)
* Evaluate with **MAE**, **RMSE**, and **R² score**

---

