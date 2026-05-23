# Abstract

This study presents an experimental platform that integrates computer vision and machine learning to support approximate pH estimation and endpoint detection in titration experiments for science education. A Raspberry Pi-based setup was used to capture real-time solution images, which were converted into RGB data for analysis. Grid-based image preprocessing reduced artifacts caused by ripples and localized color variations. Cluster analysis identified three RGB-based solution categories that were correlated with pH. Regression analysis, including Random Forest modeling, achieved high predictive accuracy with low error. Machine learning classification models were also evaluated, with Random Forest and K-Nearest Neighbors showing strong performance for the non-linear relationship between pH and RGB values. The results support the feasibility of using BTB within its transition range for approximate pH estimation and endpoint detection in an educational setting. The system can also be used as an educational platform through which students engage with automated data collection, machine learning, and real-time analysis. By reducing subjective visual observation and improving experimental reproducibility, this approach supports the use of digital technologies in science education.

# 1. Introduction

The use of digital technologies in education has expanded across disciplines and has opened new possibilities for laboratory instruction [ 1 , 2 , 3 ]. In science education, computer vision, artificial intelligence, and machine learning can bridge theoretical concepts and practical experimentation while also fostering computational thinking. These technologies enable automated data collection, reduce reliance on subjective observation, and provide real-time feedback during laboratory activities.

The color change of indicators is a fundamental concept widely utilized across various scientific fields, including chemistry, neuroscience, and medicine. Indicators are substances that exhibit a visible change, often a color shift, in response to specific chemical, biological, or physical conditions. This property makes them invaluable tools for detecting, measuring, and visualizing various analytes or physiological changes. In chemistry, indicators are essential for qualitative and quantitative analyses. They help detect the presence and concentration of specific ions or molecules by exhibiting a distinct color change in response to chemical reactions or changes in pH or redox potential [ 4 ]. Indicators in neuroscience play an important role in visualizing and measuring neuronal activity to study complex brain functions, and indicators in medicine facilitate diagnostics by providing visual signals corresponding to physiological or biochemical changes [ 5 , 6 ].

Artificial intelligence has advanced digital colorimetry by enabling camera-based quantification of analytes. In smartphone and webcam-based workflows, machine learning models process assay images, extract color features, and map them to concentration or pH with low error. Denoising and illumination normalization further improve reliability [ 7 , 8 , 9 , 10 ]. In related domains, AI is used through different optical mechanisms. In neuroscience, for example, AI analyzes fluorescent reporters whose intensity or spectrum varies with activity, enabling automated segmentation and spike inference rather than true colorimetric change [ 11 , 12 , 13 , 14 ]. In medicine, AI-assisted applications interpret test-strip images to provide rapid at-home readouts within validated ranges [ 15 , 16 ]. Beyond these domains, color-based machine vision combined with white-box machine learning has also been applied to food-quality assessment, demonstrating the broad applicability of camera-based color analysis [ 17 ].

Recent studies on automated titration monitoring have shown that computer vision can be used for endpoint detection and pH estimation. Boppana et al. [ 18 ] developed a low-cost automated titration system using colorimetric endpoint detection, while Kosenkov and Kosenkov [ 19 ] demonstrated automatic titration using computer vision techniques. More recent work has extended this area through a review of AI-assisted colorimetric detection [ 20 ] and robotic colorimetric titration platforms with computer vision-based real-time monitoring [ 21 ]. However, these prior approaches differ from the present work in important respects. Many rely on sampling the reacting solution and transferring it to a separate cuvette or indicator strip for off-line analysis, or they focus exclusively on binary endpoint detection rather than approximate continuous pH estimation. In addition, most are designed for analytical laboratory use and do not address educational implementation. In contrast, this study directly images the reacting liquid within the titration vessel in real time, extracts RGB features without requiring separate sampling, and applies both regression and classification models to estimate pH across the full BTB transition range. The entire workflow—from hardware assembly through model training and evaluation—is organized as a single, low-cost laboratory activity for science education, enabling students to integrate chemistry, programming, and data analysis in one session.

# 2. Theoretical Background

## 2.1. Colorimetric Analysis of pH Indicators

Direct measurement of pH in various materials has long been a challenge for scientists. Craig [ 22 ] introduced a method for measuring aerosol acidity using pH-indicator papers combined with RGB-based colorimetric analysis, including models such as R/G vs. pH and G-B vs. pH. Building on this approach, Li [ 23 ] proposed a linear model relating RGB values to the predicted pH (pH predict ). Because a color is represented by a combination of R (red), G (green), and B (blue) values, a linear combination of these three primary colors can capture color characteristics and, in turn, reflect the pH associated with that color. To account for changes in light intensity, each color channel is normalized using the following equations: R normal = R /( R + G + B ) G normal = G /( R + G + B ) B normal = B /( R + G + B )

The linear model is then expressed as pH predict = aR normal + bG normal + cB normal , where a, b, and c are coefficients. A color chart is typically used as a reference for pH measurements with indicator papers. By performing linear regression on the color chart, one can estimate the coefficient vector [a, b, c], which can then be used to predict the pH (pH predict ) of unknown samples.

Khanal [ 7 ] evaluated combinations of machine learning models and image color spaces, including RGB, HSV, and LAB, for predicting analyte concentrations. Models that used both sample color and reference color showed improved predictive performance. Elsenety [ 24 ] applied machine learning techniques to common pH paper for precise pH estimation, with a K-Neighbors Regressor model achieving an R 2 value of 0.995. Hou [ 25 ] reported a simple and reliable method for single-cell pH imaging and sensing by combining UV-Vis spectroscopy with common pH indicators.

The selection of the RGB color space over alternative representations (CIELab, HSV, and XYZ) was based on several technical and practical considerations. The RGB color space offers direct correspondence with digital camera sensor output, eliminating additional color space conversion steps that may introduce computational errors and processing delays.

## 2.2. Computer Vision

Computer vision (CV) technology enables machines to interpret and understand the visual world by transforming images into numerical or symbolic information. It involves multiple steps, such as image acquisition, preprocessing, segmentation, feature extraction, and analysis [ 26 ]. CV systems utilize digital cameras, sensors, and imaging software to capture images from real-world environments. Image preprocessing techniques, such as normalization, contrast enhancement, and noise reduction, are applied to improve image quality and ensure consistent analysis. The integration of AI and CV has led to significant advancements in areas such as object detection, facial recognition, and autonomous vehicles.

In the context of colorimetric analysis, AI and computer vision technologies have been used to improve the interpretation of indicator-based colorimetric data [ 24 , 25 ]. By training machine learning models on large image datasets, researchers can model the relationship between RGB values and chemical properties such as pH [ 7 ]. These advances have also supported automated endpoint detection in titration experiments [ 18 ]. In the present study, the color change of Bromothymol Blue (BTB) during titration is analyzed using a computer vision system combined with machine learning. This setup enables more objective, accurate, and reproducible endpoint detection, thereby improving the quality and reliability of the experimental results [ 19 ].

## 2.3. Digital Technology in Science Education

The application of digital technology in education has gained significant momentum, driven by advances in affordable computing platforms, open-source software, and machine learning frameworks [ 1 , 2 , 3 ]. In science education, technology-enhanced learning environments provide students with opportunities to engage in authentic scientific inquiry, develop data literacy skills, and experience the interdisciplinary nature of modern research. The integration of computational tools into laboratory settings transforms passive observation into active, data-driven experimentation, aligning with contemporary educational frameworks that emphasize STEM competencies and 21st-century skills.

Single-board computers such as the Raspberry Pi have emerged as practical educational platforms that enable hands-on learning in programming, electronics, and data analysis at minimal cost. When combined with computer vision libraries (e.g., OpenCV) and machine learning frameworks (e.g., scikit-learn), these platforms offer a workable environment for developing technology-enhanced experiments that maintain both educational value and scientific rigor. Such setups respond to the growing need for teaching methods that incorporate digital tools, as noted in recent educational technology literature.

# 3. Materials and Methods

The development of the automated pH prediction system followed a structured methodology comprising three phases: (1) hardware assembly and experimental device development, (2) titration and data collection using BTB solutions, and (3) data analysis and machine learning model evaluation. Each phase was designed to be replicable in educational laboratory settings using commercially available, low-cost components.

## 3.1. Hardware Setup

The experimental device consists of a Raspberry Pi, a webcam, a monitor, a power supply, and a keyboard; the circuit diagram is shown in Figure 1 . This hardware configuration was chosen for its suitability in educational settings, where cost-effectiveness and ease of assembly are important. The Raspberry Pi served as the central processing unit for real-time image capture and data processing. The webcam was the image input device for collecting RGB data, and a USB keyboard was used for command input and system control. A dedicated power adapter ensured the stable operation of the Raspberry Pi.

The equipment specifications are as follows: Raspberry Pi 4 Model B (8 GB RAM, Broadcom BCM2711 quad-core ARM Cortex-A72 @ 1.5 GHz; Raspberry Pi Ltd., Cambridge, UK), Logitech C920 HD Pro webcam(1920 × 1080 pixels, 30 fps; Logitech Europe S.A., Lausanne, Switzerland), Raspberry Pi 7-inch Touch Display (800 × 480 pixels; Raspberry Pi Ltd., Cambridge, UK), USB keyboard compatible with the Raspberry Pi USB interface (Logitech Europe S.A., Lausanne, Switzerland), and Raspberry Pi 15W USB-C Power Supply (5.1 V, 3.0 A DC; Raspberry Pi Ltd., Cambridge, UK).

## 3.2. Titration of BTB Solution

The experimental setup is shown in Figure 2 . Sample solutions with pH values ranging from approximately 4.50 to 11.00 were prepared, and BTB indicator was added to each solution. Color images were captured using a webcam and converted into RGB data, which were then used to train the machine learning models. Accurate pH values were measured simultaneously using a calibrated digital pH meter.

### 3.2.1. Preparation of Standard Buffer Solutions

Reagent-grade pH 4.00, 7.00, and 9.18 standard buffer solutions were prepared at 25 °C. For the pH 4.00 buffer, 10.21 g of potassium hydrogen phthalate (KHC 8 H 4 O 4 ) was accurately weighed, dissolved in 800 mL of ultrapure water, transferred to a 1 L volumetric flask, and diluted to the mark with ultrapure water. For the pH 7.00 buffer, 3.39 g of potassium dihydrogen phosphate (KH 2 PO 4 ) and 3.55 g of disodium hydrogen phosphate (Na 2 HPO 4 ) were accurately weighed and prepared in the same manner. For the pH 9.18 buffer, 3.81 g of sodium tetraborate decahydrate (Na 2 B 4 O 7 ·10H 2 O) was accurately weighed and prepared in the same manner. All buffer solutions were prepared on the day of use and kept sealed until calibration to minimize CO 2 absorption and pH drift.

### 3.2.2. Calibration Steps

The pH electrode was rinsed with ultrapure water and gently blotted dry before calibration. The electrode was immersed in the pH 7.00 buffer solution until the reading stabilized. This process was repeated with the pH 4.00 and 9.18 buffers. The calibration cycle was repeated until the readings in all buffer solutions stabilized within ±0.02 pH units of their known values.

### 3.2.3. Extraction of Color Data from Titration Experiments

BTB powder was dissolved completely in ethanol (20 mL), and distilled water was added (80 mL) to bring the total solution volume to 100 mL. Approximately 60 μL of BTB solution was added to 3–5 mL of pH 7.0 buffer solution. 50 mL of pH 7.0 BTB solution was added to an Erlenmeyer flask and titrated with 0.1 M NaOH solution while monitoring color changes.

Color images of solutions during the acid-base titration process were captured using a webcam, and the image data were converted to RGB data. Strict control over the surrounding illumination conditions was implemented. Illumination was provided by two LED panels with a correlated color temperature of 5600 K. Camera settings were standardized by disabling automatic white balance. Samples were placed against a matte white background board to reduce secondary optical artifacts.

## 3.3. Data Analysis and Prediction

The study proceeded through the following main stages: data collection, preprocessing, cluster analysis, regression analysis, and classification-model evaluation. These stages were used to analyze and predict solution color changes by combining computer vision with machine learning, as summarized in Figure 3 [ 26 ].

In the data collection phase, Bromothymol Blue (BTB) indicator was added to various acidic, basic, or neutral solutions to induce color changes according to different pH values. A high-resolution webcam was used to capture images of the solutions while simultaneously measuring the exact pH value of each sample with a pH meter.

The preprocessing stage organized the collected image data for use in the machine learning analysis [ 27 ]. The original frames (4032 × 3024 pixels; 54 images) were first center-cropped to a square format (3024 × 3024 pixels) to standardize the field of view and remove edge artifacts. Each cropped frame was then divided into 336 × 336 pixel patches, producing an augmented patch-level dataset of 4374 images [ 28 ]. The images were processed in OpenCV through region definition and cropping, color conversion (BGR to RGB), denoising, illumination normalization, patch extraction, and per-patch mean computation. Feature vectors consisted of the patch-averaged R, G, and B values together with the normalized channel values. Pseudocode is provided in Appendix A .

To ensure the independence between the training and test sets, the data splitting was performed at the original image level before patch extraction. The 54 original images were first randomly divided into training (43 images, 80%) and test (11 images, 20%) sets. Patches were then extracted independently from each set, producing 3499 training patches and 875 test patches. This image-level splitting strategy prevents data leakage that could arise if patches from the same original image appeared in both training and test sets, thereby providing a more rigorous evaluation of model generalization. The titration experiments were conducted across 6 independent experimental sessions over a 4-week period, with each session producing 8–10 images at different pH levels to ensure experimental reproducibility.

The complete image acquisition and ROI-extraction pipeline is illustrated in Figure 4 . K-means clustering was applied to classify the images into three distinct clusters in RGB feature space [ 29 ]. Multiple linear regression analysis was then used to quantify the relationship between RGB values and pH [ 30 , 31 ]. Several machine learning classification models, including K-Nearest Neighbors (KNN), Random Forest, Gradient Boosting, AdaBoost, and Support Vector Machine (SVM), were implemented and evaluated [ 32 , 33 , 34 , 35 , 36 , 37 ]. Model performance was assessed using cross-validation together with accuracy, precision, recall, and F1 score metrics [ 38 , 39 ].

# 4. Results

## 4.1. Clustering Analysis Results

The clustering results shown in Figure 5 indicate that the dataset can be separated into distinct groups according to the characteristics of the RGB variables. Each cluster exhibits a specific color-profile pattern, indicating a clear relationship with pH distribution. The elbow method was used to determine the optimal number of clusters, and the curve shows a clear elbow at k = 3.

The detailed clustering results are summarized in Table 1 . The cluster boundaries identified by the K-means algorithm (pH 6.72 and pH 7.71) closely correspond to the theoretical transition zone of the BTB indicator (pH 6.0–7.6). This alignment is chemically meaningful: below pH 6.72, BTB exists predominantly in its protonated (yellow) form with high R values and low B values; between pH 6.72 and 7.71, the indicator undergoes its characteristic yellow-to-blue transition, producing intermediate green hues; and above pH 7.71, the deprotonated (blue) form dominates with high B values and negligible R values, as shown in Figure 6 . The correspondence between data-driven cluster boundaries and the known BTB transition range validates the chemical basis of the RGB-based classification approach.

## 4.2. Multiple Regression Analysis Results

The multiple regression analysis compared four models, namely Linear Regression, Decision Tree Regression, Random Forest Regression, and Support Vector Regression, as shown in Table 2 . Among these, the Random Forest model showed the best performance, with an R 2 value greater than 0.85 on the test set and the lowest MSE and MAE values, as shown in Table 3 .

The moderate R 2 values observed across the regression models reflect the inherent difficulty of predicting exact pH values from RGB data within BTB’s narrow transition range (pH 6.0–7.6). Within this interval, subtle color gradations correspond to small pH differences, making precise regression more challenging than categorical classification. Even so, the Random Forest model achieved an MAE below 0.3 pH units, as shown in Figure 7 , which is adequate for the educational titration applications that are the primary focus of this study.

## 4.3. Classification Model Results

In the classification analysis, three classes were defined using threshold values informed by the K-means clustering results: acidic (pH 4.50–6.13), neutral (pH 6.26–7.62), and basic (pH 7.71–11.00). This three-class scheme was motivated by both data-driven evidence and educational relevance. From a data-driven perspective, the K-means clustering analysis in Section 4.1 identified three naturally separable RGB clusters, with the elbow plot of within-cluster inertia showing a clear inflection at k = 3, confirming that three clusters best capture the structure in the RGB feature space. From an educational perspective, the three classes correspond directly to the acidic–neutral–basic categories that students learn in standard science curricula, making the classification output immediately interpretable in a teaching context. These three categories also map onto the well-known color-transition zones of the BTB indicator (yellow, green, and blue). It should be noted that this study also provides continuous pH estimation through regression models ( Section 4.2 ); the three-class classification therefore complements, rather than replaces, the regression analysis by offering a rapid categorical assessment of the solution state. Both K-Nearest Neighbors (KNN) and Random Forest achieved high accuracy ( Table 4 ). However, this high accuracy is largely attributable to the well-separated color transitions of BTB, which produce clearly distinguishable RGB clusters; the task would be more challenging for indicators with less distinct or overlapping color changes. Random Forest also showed stable predictive performance when class boundaries were less sharply defined, as shown in Table 5 .

The feature-importance analysis of the Random Forest model showed that the R channel was the most influential predictor (approximately 40.7%), followed by the G channel (35.6%) and the B channel (23.7%).

The perfect classification accuracy (100%) reported in Table 4 reflects performance on the held-out test set and requires careful interpretation. This result is likely attributable to the well-separated BTB color clusters observed across different pH ranges, as illustrated in Figure 5 . BTB produces highly distinguishable colors, yellow for acidic, green for neutral, and blue for basic solutions, which form clearly separated groups in RGB feature space. To further assess the possibility of overfitting, 5-fold cross-validation was additionally performed on the training set, yielding consistent mean accuracies of 99.5 ± 0.4% for Random Forest and 99.3 ± 0.6% for KNN across all folds. In addition, the image-level train/test split described in Section 3.3 ensured that training and test patches originated from different images, thereby reducing the risk of data leakage. Nevertheless, the relatively small number of original images and the patch-based augmentation strategy may still inflate performance metrics.

# 5. Discussion

Previous studies have mainly analyzed indicator color using pH strips [ 40 , 41 ], while related recent work has focused on endpoint detection, AI-assisted colorimetric sensing, or titration automation [ 18 , 20 , 21 ]. In contrast, the present study estimates pH directly from real-time images of the reaction solution and integrates hardware assembly, image processing, and machine learning within a single educational laboratory activity. This distinguishes our approach from most existing systems, which are typically designed for analytical laboratory use rather than educational implementation.

The color change of BTB is well suited for endpoint indication and for approximate pH estimation within its transition interval, but the resulting RGB-based estimates are intended for the educational scope of this study and should not be interpreted as a substitute for direct, high-precision pH measurement.

Bromothymol Blue (BTB) was selected because its color transitions within the pH 6.0–7.6 range produce measurable RGB shifts across the acidic-to-basic spectrum. Among the regression models tested, Random Forest yielded the best predictive performance. In the classification task, both KNN and Random Forest reached 100% accuracy on the present dataset, an outcome attributable to the well-separated yellow/green/blue clusters of BTB; for indicators with more gradual or continuous color shifts, achieving comparable accuracy would likely require more advanced feature representations or model architectures. The Random Forest feature-importance ranking showed that the R channel contributed the most (40.7%), followed by the G channel (35.6%) and the B channel (23.7%), consistent with BTB’s color transition. Because image-level splitting was applied to separate training and test sets, direct data leakage is mitigated; however, patches originating from the same titration session may share background and illumination characteristics, and this should be taken into account when interpreting the reported accuracy.

## 5.1. Educational Implications and Applications

This study is relevant to the theme of the special issue because it demonstrates one approach to integrating digital technology into science education. The system uses low-cost hardware and open-source software to transform a routine titration experiment into a laboratory activity in which students engage simultaneously with chemistry, programming, and data analysis.

From a cost perspective, the RGB-based pH-estimation approach requires only a webcam and a Raspberry Pi, with total hardware costs of approximately USD 100–150. This cost level is feasible even for schools with limited laboratory budgets. During the experiment, students assemble the hardware, write Python (3.12) scripts for image capture and preprocessing, and train machine learning models—all within a single laboratory session.

Automated endpoint detection also reduces the subjectivity associated with visual color judgment, which is a common source of error in student titrations. Because the system records each frame together with its RGB values, instructors can review the data with students and discuss differences between human observation and algorithmic decisions.

The setup also supports open-ended investigation. For example, students can examine how lighting conditions affect RGB values, replace BTB with another indicator, and compare model performance or vary the grid size used in averaging and observe the effect on prediction error. These activities move beyond a fixed protocol and support a cycle of hypothesis, experiment, and evaluation.

Extending the training data to include other common indicators, such as phenolphthalein and methyl orange, would broaden the range of titrations that the system can support. Connecting the Raspberry Pi to a shared server or cloud notebook could also allow multiple student groups to pool their data, thereby increasing dataset size and supporting collaborative data practices.

## 5.2. Limitations

Several limitations of this study should be acknowledged. First, the dataset was constructed from 54 original titration images, which were divided into patches to create an augmented dataset of 4374 samples. Although the image-level data splitting mitigates direct data leakage, patches derived from a single titration run may share illumination and background characteristics, potentially inflating model performance relative to fully independent samples. Future work should therefore include independently collected validation datasets to strengthen the assessment of model generalization.

Second, the experimental validation is restricted to a single indicator (BTB) with a relatively narrow pH working range (pH 6.0–7.6). Other indicators, such as phenolphthalein (pH 8.2–10.0) or methyl orange (pH 3.1–4.4), exhibit different color transitions and pH ranges, and the applicability of the current approach to these indicators has not been tested. A related concern is that the high classification accuracy reported here is closely tied to the clear color separation of BTB, which simplifies the classification task. For indicators with less distinct, gradual, or overlapping color changes across pH levels, the same level of accuracy may not be achievable, and additional feature engineering or more complex models may be required. The developed framework should therefore be considered a proof of concept rather than a general analytical method, and extending it to other indicators is a necessary next step for future research.

Third, RGB-based colorimetric analysis is inherently sensitive to illumination conditions. Although controlled lighting was employed in this study ( Section 3.2.3 ), variations in ambient lighting, camera sensor characteristics, and white-balance settings could reduce the reproducibility of RGB measurements across different experimental setups. The system’s performance under varying environmental conditions has not yet been systematically evaluated, and future work should test the framework with different lighting environments, imaging devices, and indicator types to establish broader applicability.

Overall, this study demonstrates how low-cost hardware and standard machine learning models can be used in an educational laboratory context. Its main contribution lies in the practical educational framework rather than in methodological advances in analytical chemistry or machine learning.

Future research should examine indicators with wider pH ranges in order to move beyond the relatively narrow working range of BTB and should also develop adaptive normalization strategies for use under varying environmental conditions.

Although the present study is limited to BTB and a relatively small set of 54 original images, the framework is extensible to other indicators and larger datasets. The patch-based augmentation strategy was effective for increasing sample size, but it may also introduce correlations among training samples; this issue should be addressed in future work through additional independently collected experimental data.

# 6. Conclusions

In this study, computer vision and machine learning were combined to support approximate pH estimation and endpoint detection in BTB-based titration. A Raspberry Pi captured solution images in real time, and grid-based averaging of RGB patches reduced noise caused by ripples and localized color variation. Among the regression models tested, Random Forest performed best (R 2 > 0.85, MAE < 0.3 pH units). For three-class classification (acidic, neutral, and basic), both Random Forest and KNN reached perfect accuracy on the test set, although these results should be interpreted within the proof-of-concept scope and the data limitations discussed above.

Because the entire pipeline, from hardware setup through model evaluation, runs on inexpensive open-source components, it can be adopted in a wide range of school and university settings. A single laboratory activity allows students to work with analytical chemistry, image processing, and machine learning within the same experimental framework. Because patches within a single titration run may share background and illumination properties, future evaluations should include independently collected datasets. Testing the framework under varied lighting, with different imaging devices, and with indicators other than BTB will be necessary to confirm the generalizability of the approach.

# Appendix A. Pseudocode

# References

- Daniela, L. Pedagogies of Digital Learning in Higher Education ; Routledge: London, UK, 2020. [ Google Scholar ]

- Selwyn, N. Education and Technology: Key Issues and Debates , 3rd ed.; Bloomsbury Academic: London, UK, 2022. [ Google Scholar ]

- OECD. OECD Digital Education Outlook 2023: Towards an Effective Digital Education Ecosystem ; OECD Publishing: Paris, France, 2023. [ Google Scholar ]

- Steinegger, A.; Wolfbeis, O.S.; Borisov, S.M. Optical sensing and imaging of pH values: Spectroscopies, materials, and applications. Chem. Rev. 2020 , 120 , 12357–12489. [ Google Scholar ] [ CrossRef ]

- Piatkevich, K.D.; Murdock, M.H.; Subach, F.V. Advances in engineering and application of optogenetic indicators for neuroscience. Appl. Sci. 2019 , 9 , 562. [ Google Scholar ] [ CrossRef ]

- Bhatia, D.; Paul, S.; Acharjee, T.; Ramachairy, S.S. Biosensors and their widespread impact on human health. Sens. Int. 2024 , 5 , 100257. [ Google Scholar ] [ CrossRef ]

- Khanal, B.; Pokhrel, P.; Khanal, B.; Giri, B. Machine-learning-assisted analysis of colorimetric assays on paper analytical devices. ACS Omega 2021 , 6 , 33837–33845. [ Google Scholar ] [ CrossRef ]

- Chen, Y.; Fu, Q.; Li, D.; Xie, J.; Ke, D.; Song, Q.; Tang, Y.; Wang, H. A smartphone colorimetric reader integrated with an ambient light sensor and a 3D printed attachment for on-site detection of zearalenone. Anal. Bioanal. Chem. 2017 , 409 , 6567–6574. [ Google Scholar ] [ CrossRef ] [ PubMed ]

- Soares, S.; Fernandes, G.M.; Rocha, F.R. Smartphone-based digital images in analytical chemistry: Why, when, and how to use. TrAC Trends Anal. Chem. 2023 , 168 , 117284. [ Google Scholar ] [ CrossRef ]

- Das, A.J.; Wahi, A.; Kothari, I.; Raskar, R. Ultra-portable, wireless smartphone spectrometer for rapid, non-destructive testing of fruit ripeness. Sci. Rep. 2016 , 6 , 32504. [ Google Scholar ] [ CrossRef ]

- Pnevmatikakis, E.A.; Giovannucci, A. NoRMCorre: An online algorithm for piecewise rigid motion correction of calcium imaging data. J. Neurosci. Methods 2017 , 291 , 83–94. [ Google Scholar ] [ CrossRef ]

- Apthorpe, N.; Riordan, A.; Aguilar, R.; Homann, J.; Gu, Y.; Tank, D.; Seung, H.S. Automatic neuron detection in calcium imaging data using convolutional networks. Adv. Neural Inf. Process. Syst. 2016 , 29 . [ Google Scholar ]

- Soltanian-Zadeh, S.; Sahingur, K.; Blau, S.; Gong, Y.; Farsiu, S. Fast and robust active neuron segmentation in two-photon calcium imaging using spatiotemporal deep learning. Proc. Natl. Acad. Sci. USA 2019 , 116 , 8554–8563. [ Google Scholar ] [ CrossRef ] [ PubMed ]

- Stringer, C.; Pachitariu, M.; Steinmetz, N.; Reddy, C.B.; Carandini, M.; Harris, K.D. Spontaneous behaviors drive multidimensional, brainwide activity. Science 2019 , 364 , eaav7893. [ Google Scholar ] [ CrossRef ]

- Mudanyali, O.; Dimitrov, S.; Sikora, U.; Padmanabhan, S.; Navruz, I.; Ozcan, A. Integrated rapid-diagnostic-test reader platform on a cellphone. Lab. Chip 2012 , 12 , 2678–2686. [ Google Scholar ] [ CrossRef ]

- Contreras, I.; Vehi, J. Artificial intelligence for diabetes management and decision support: Literature review. J. Med. Internet Res. 2018 , 20 , e10775. [ Google Scholar ] [ CrossRef ]

- Sánchez, C.N.; Orvañanos-Guerrero, M.T.; Domínguez-Soberanes, J.; Álvarez-Cisneros, Y.M. Analysis of beef quality according to color changes using computer vision and white-box machine learning techniques. Heliyon 2023 , 9 , e17976. [ Google Scholar ] [ CrossRef ]

- Boppana, N.P.D.; Snow, R.; Simone, P.S.; Emmert, G.L.; Brown, M.A. A low-cost automated titration system for colorimetric endpoint detection. Analyst 2023 , 148 , 2133–2140. [ Google Scholar ] [ CrossRef ]

- Kosenkov, Y.; Kosenkov, D. Computer Vision in Chemistry: Automatic Titration. J. Chem. Educ. 2021 , 98 , 4067–4073. [ Google Scholar ] [ CrossRef ]

- Parakh, A.; Awate, A.; Barman, S.M.; Kadu, R.K.; Tulaskar, D.P.; Kulkarni, M.B.; Bhaiyya, M. Artificial intelligence and machine learning for colorimetric detections: Techniques, applications, and future prospects. Trends Environ. Anal. Chem. 2025 , 48 , e00280. [ Google Scholar ] [ CrossRef ]

- Li, Y.; Dutta, B.; Yeow, Q.J.; Clowes, R.; Boott, C.E.; Cooper, A.I. High-throughput robotic colourimetric titrations using computer vision. Digit. Discov. 2025 , 4 , 1276–1283. [ Google Scholar ] [ CrossRef ]

- Craig, R.L.; Peterson, P.K.; Nandy, L.; Lei, Z.; Hossain, M.A.; Camarena, S.; Dodson, R.A.; Cook, R.D.; Dutcher, C.S.; Ault, A.P. Direct Determination of Aerosol pH: Size-Resolved Measurements. Anal. Chem. 2018 , 90 , 11232–11239. [ Google Scholar ] [ CrossRef ] [ PubMed ]

- Li, G.; Su, H.; Ma, N.; Zheng, G.; Kuhn, U.; Li, M.; Klimach, T.; Pöschl, U.; Cheng, Y. Multifactor colorimetric analysis on pH-indicator papers: An optimized approach for direct determination of ambient aerosol pH. Atmos. Meas. Tech. Discuss. 2020 , 13 , 6053–6065. [ Google Scholar ] [ CrossRef ]

- Elsenety, M.M.; Mohamed, M.B.I.; Sultan, M.E.; Elsayed, B.A. Facile and highly precise pH-value estimation using common pH paper based on machine learning techniques and supported mobile devices. Sci. Rep. 2022 , 12 , 22584. [ Google Scholar ] [ CrossRef ]

- Hou, H.; Zhao, Y.; Li, C.; Wang, M.; Xu, X.; Jin, Y. Single-cell pH imaging and detection for pH profiling and label-free rapid identification of cancer-cells. Sci. Rep. 2017 , 7 , 1759. [ Google Scholar ] [ CrossRef ]

- Prakash, D.C.; Narayanan, R.; Ganesh, N.; Ramachandran, M.; Chinnasami, S.; Rajeshwari, R. A study on image processing with data analysis. AIP Conf. Proc. 2022 , 2393 , 020225. [ Google Scholar ]

- Géron, A. Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow ; O’Reilly Media: Sebastopol, CA, USA, 2019. [ Google Scholar ]

- Raschka, S.; Liu, Y.H.; Mirjalili, V. Machine Learning with PyTorch and Scikit-Learn ; Packt Publishing: Birmingham, UK, 2022. [ Google Scholar ]

- James, G.; Witten, D.; Hastie, T.; Tibshirani, R.; Taylor, J. Statistical Learning. In An Introduction to Statistical Learning: With Applications in Python ; Springer: Berlin/Heidelberg, Germany, 2023; pp. 15–67. [ Google Scholar ]

- Subramanya, S.M.; Rios, N.; Kollar, A.; Stofanak, R.; Maloney, K.; Waltz, K.; Powers, L.; Rane, C.; Savage, P.E. Statistical models for predicting oil composition from hydrothermal liquefaction of biomass. Energy Fuels 2023 , 37 , 6619–6628. [ Google Scholar ] [ CrossRef ]

- Lafuente, D.; Cohen, B.; Fiorini, G.; García, A.A.; Bringas, M.; Morzan, E.; Onna, D. A gentle introduction to machine learning for chemists: An undergraduate workshop using Python notebooks for visualization, data processing, analysis, and modeling. J. Chem. Educ. 2021 , 98 , 2892–2898. [ Google Scholar ] [ CrossRef ]

- Hodson, T.O. Root-mean-square error (RMSE) or mean absolute error (MAE): When to use them or not. Geosci. Model Dev. 2022 , 15 , 5481–5487. [ Google Scholar ] [ CrossRef ]

- Müller, A.C.; Guido, S. Introduction to Machine Learning with Python: A Guide for Data Scientists ; O’Reilly Media: Sebastopol, CA, USA, 2016. [ Google Scholar ]

- Cunningham, P.; Delany, S.J. K-nearest neighbour classifiers—A tutorial. ACM Comput. Surv. 2021 , 54 , 128. [ Google Scholar ] [ CrossRef ]

- Breiman, L. Random forests. Mach. Learn. 2001 , 45 , 5–32. [ Google Scholar ] [ CrossRef ]

- Freund, Y.; Schapire, R.E. A decision-theoretic generalization of on-line learning and an application to boosting. J. Comput. Syst. Sci. 1997 , 55 , 119–139. [ Google Scholar ] [ CrossRef ]

- Cortes, C.; Vapnik, V. Support-Vector Networks. Mach. Learn. 1995 , 20 , 273–297. [ Google Scholar ] [ CrossRef ]

- Powers, D.M. Evaluation: From precision, recall and F-measure to ROC, informedness, markedness and correlation. arXiv 2020 , arXiv:2010.16061. [ Google Scholar ] [ CrossRef ]

- Fawcett, T. An introduction to ROC analysis. Pattern Recognit. Lett. 2006 , 27 , 861–874. [ Google Scholar ] [ CrossRef ]

- Sri Sruthi, P.; Balasubramanian, S.; Senthil Kumar, P.; Kapoor, A.; Ponnuchamy, M.; Mariam Jacob, M.; Prabhakar, S. Eco-friendly pH detecting paper-based analytical device: Towards process intensification. Anal. Chim. Acta 2021 , 1182 , 338953. [ Google Scholar ] [ CrossRef ] [ PubMed ]

- Pastore, A.; Badocco, D.; Bogialli, S.; Cappellin, L.; Pastore, P. pH Colorimetric Sensor Arrays: Role of the color space adopted for the calculation of the prediction error. Sensors 2020 , 20 , 6036. [ Google Scholar ] [ CrossRef ]

# FiguresandTables

# html-copyright