---
title: Lung Cancer Risk Assessment Tool
emoji: 🫁
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🫁 Lung Cancer Risk Assessment Tool

An AI-powered risk assessment tool for lung cancer using real medical literature data from PubMed.

## 🚀 Features

### 📄 Text Analysis
- Analyze medical notes, clinical reports, or patient presentations
- NLP-based risk assessment using TF-IDF vectorization
- Trained on 2,382 real PubMed abstracts
- Provides risk level, probability score, and clinical interpretation

### 🏥 Clinical Data Analysis
- Comprehensive risk assessment using patient demographics
- Laboratory values (CEA, hemoglobin, WBC count)
- Symptom analysis (weight loss, cough, dyspnea, chest pain)
- 5-level risk stratification with clinical recommendations

## 🔬 Model Details

### Data Source
- **Training Data**: 2,382 real medical abstracts from PubMed
- **Cancer Cases**: 882 lung cancer abstracts
- **Control Cases**: 1,500 non-cancer respiratory condition abstracts
- **Collection Method**: Enhanced PubMed API with duplicate removal

### Machine Learning
- **Text Model**: Logistic Regression with L1/L2 regularization
- **Clinical Model**: Multi-algorithm comparison (Random Forest, Gradient Boosting, Logistic Regression)
- **Features**: 40 engineered clinical features
- **Validation**: 10-fold cross-validation with overfitting detection

### Performance
- **Text Model AUC**: 1.000 (with validation warnings)
- **Clinical Model AUC**: 0.991
- **Sensitivity**: 0.936
- **Specificity**: 0.944

## ⚠️ Important Disclaimers

**This is a research prototype for educational purposes only.**

- 🚫 **Not a diagnostic tool** - Do not use for medical diagnosis
- 👨‍⚕️ **Consult healthcare professionals** - Always seek professional medical advice
- 🔬 **Research use only** - Intended for academic and educational purposes
- 📊 **Validation needed** - Requires clinical validation before any medical application

## 🎯 Use Cases

### Educational
- Medical student training
- Healthcare education
- AI/ML demonstrations
- Research methodology examples

### Research
- Benchmarking other models
- Feature importance analysis
- Clinical decision support research
- NLP in healthcare studies

## 📖 How to Use

### Text Analysis
1. Enter medical text in the "Text Analysis" tab
2. Include clinical findings, patient presentation, or medical notes
3. Click "Analyze Text" to get risk assessment
4. Review probability score and clinical interpretation

### Clinical Assessment
1. Go to "Clinical Data Analysis" tab
2. Enter patient demographics (age, sex, smoking history)
3. Add laboratory values (CEA, hemoglobin, WBC)
4. Select relevant symptoms
5. Click "Assess Risk" for comprehensive analysis

## 🔧 Technical Implementation

### Architecture
```
Input → Preprocessing → Feature Extraction → ML Model → Risk Assessment
```

### Text Processing Pipeline
1. **Text Cleaning**: Remove research bias and statistical jargon
2. **Tokenization**: Extract relevant medical terms
3. **Vectorization**: TF-IDF with medical stop words
4. **Classification**: Regularized logistic regression

### Clinical Processing Pipeline
1. **Feature Engineering**: 40 clinical features including interactions
2. **Normalization**: StandardScaler for numerical features
3. **Risk Stratification**: Multi-threshold probability mapping
4. **Recommendations**: Evidence-based clinical guidelines

## 🛡️ Model Robustness

### Overfitting Prevention
- ✅ Regularization (L1/L2 penalties)
- ✅ Cross-validation with multiple folds
- ✅ Automated validation warnings
- ✅ Advanced duplicate removal
- ✅ Quality filtering of training data

### Validation Checks
- CV score consistency monitoring
- AUC threshold validation
- Train/test performance gap detection
- Automated warning system

## 📊 Risk Levels

### Text Analysis
- 🔴 **HIGH RISK** (>0.8): Strong malignancy indicators
- 🟡 **MODERATE-HIGH** (0.6-0.8): Multiple concerning features
- 🟡 **MODERATE** (0.4-0.6): Some concerning features
- 🟢 **LOW RISK** (<0.4): Suggests benign condition

### Clinical Assessment
- 🔴 **VERY HIGH** (>0.85): Immediate oncology referral
- 🔴 **HIGH** (0.7-0.85): Urgent specialist consultation
- 🟡 **MODERATE-HIGH** (0.5-0.7): Priority imaging and consultation
- 🟡 **MODERATE** (0.3-0.5): Close monitoring required
- 🟢 **LOW** (<0.3): Routine screening appropriate

## 📚 References

### Data Sources
- PubMed/MEDLINE database (NCBI)
- Peer-reviewed medical literature
- Enhanced E-utilities API queries

### Methodologies
- Text mining and NLP in healthcare
- Clinical decision support systems
- Machine learning in medical diagnosis
- Cross-validation and model validation

## 🤝 Contributing

This is a research prototype. For improvements or collaboration:
1. Fork the repository
2. Create feature branch
3. Submit pull request with detailed description
4. Ensure all changes maintain educational/research focus

## 📄 License

MIT License - See LICENSE file for details

## 🔗 Links

- [Hugging Face Space](https://huggingface.co/spaces/your-username/lung-cancer-assessment)
- [Source Code](https://github.com/your-username/lung-cancer-assessment)
- [Research Paper](https://link-to-your-paper.com) (if available)

## 📧 Contact

For questions, collaboration, or research inquiries:
- Email: your-email@domain.com
- GitHub: @your-username
- LinkedIn: your-linkedin-profile

---

**Remember: This tool is for educational and research purposes only. Always consult qualified healthcare professionals for medical decisions.**