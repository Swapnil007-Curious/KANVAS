# KANVAS
*Kolmogorov-Arnold Network Verification and Analysis Suite*

KANVAS is an intrinsically interpretable Clinical Decision Support System built on a Kolmogorov-Arnold Additive Model (KAAM). Unlike black-box neural networks or post-hoc explainability tools like SHAP and LIME, KANVAS routes each clinical variable — age, blood pressure, cholesterol — through its own independent, learnable B-spline curve. The final risk score is simply the sum of these curves passed through a sigmoid. Because every feature is mathematically decoupled, clinicians can plot the exact learned curve for any biomarker and see precisely how risk scales with that variable — non-linearly, transparently, and auditable end-to-end. Built and validated on the MIMIC-III Clinical Database, KANVAS demonstrates that interpretability and predictive accuracy need not be in conflict in medical AI.

