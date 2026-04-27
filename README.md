# Deep Learning – HW1: Linear Image Classifiers on CIFAR-10

פרויקט זה מממש שני סיווגנים לינאריים על תת-קבוצה בת 3 מחלקות של CIFAR-10:
1. **Multiclass Perceptron** – סיווגן לפי כלל הפרספטרון הקלאסי.
2. **Logistic Regression (Softmax)** – סיווגן רב-מחלקתי עם פונקציית הסתברות Softmax והפסד Cross-Entropy.

שני המודלים נאמנים עם **Stochastic Gradient Descent (SGD)** עם mini-batches, ולכל אחד מהם מבוצעת סריקת היפר-פרמטרים (`learning_rate`, `batch_size`) על סט validation.

המטרה של README זה היא להסביר באופן **מלא ומדויק** מה כל פונקציה עושה, למה היא עושה את זה, וכיצד הדבר מתקשר לחומר התיאורטי מההרצאות.

---

## תוכן עניינים

1. [מבנה הפרויקט](#מבנה-הפרויקט)
2. [רקע תיאורטי](#רקע-תיאורטי)
   - [הבעיה: סיווג רב-מחלקתי](#הבעיה-סיווג-רב-מחלקתי)
   - [מודל לינארי גנרי](#מודל-לינארי-גנרי)
   - [פרספטרון רב-מחלקתי](#פרספטרון-רב-מחלקתי)
   - [Softmax + Cross-Entropy](#softmax--cross-entropy)
   - [SGD](#stochastic-gradient-descent)
3. [זרימת הקוד (מהנתונים ועד לחיזוי)](#זרימת-הקוד)
4. [פירוט כל פונקציה בקוד](#פירוט-כל-פונקציה-בקוד)
5. [היפר-פרמטרים וסריקה](#היפר-פרמטרים-וסריקה)
6. [הוראות הרצה](#הוראות-הרצה)

---

## מבנה הפרויקט

```
HW01/
├── ex1.ipynb          # מחברת ראשית – טעינת CIFAR-10, עיבוד מקדים, אימון, הערכה, ויזואליזציות
├── linear_models.py   # ליבת הקוד: המחלקות LinearClassifier / LinearPerceptron / LogisticRegression
│                      # והפונקציות perceptron_loss_naive, softmax, softmax_cross_entropy,
│                      # softmax_cross_entropy_vectorized, tune_perceptron.
└── README.md          # המסמך הזה
```

---

## רקע תיאורטי

### הבעיה: סיווג רב-מחלקתי

נתון מדגם אימון

$$
\mathcal{D} = \\{(\mathbf{x}_i, y_i)\\}_{i=1}^{N}, \quad \mathbf{x}_i \in \mathbb{R}^{D},\\; y_i \in \\{0, 1, \ldots, C-1\\}.
$$

אצלנו, כל תמונה הופכת לוקטור באורך $D = 32 \cdot 32 \cdot 3 + 1 = 3073$ (המימד האחרון הוא קבוע 1 – ה־**bias trick**, ראו בהמשך), ו־$C = 3$ (תת־קבוצה של שלוש מחלקות ב־CIFAR-10).

המטרה היא ללמוד פונקציה $h: \mathbb{R}^{D}\to\\{0,\ldots,C-1\\}$ שמסווגת היטב דגימות חדשות.

### מודל לינארי גנרי

לכל מחלקה $c$ יש וקטור משקלות $\mathbf{w}_c \in \mathbb{R}^{D}$. נאחד אותם למטריצה

$$
W \in \mathbb{R}^{D \times C}, \qquad W = [\mathbf{w}_0 \\,|\\, \mathbf{w}_1 \\,|\\, \cdots \\,|\\, \mathbf{w}_{C-1}].
$$

עבור דגימה $\mathbf{x}$, הציונים (scores) מוגדרים כ־

$$
\mathbf{s}(\mathbf{x}) = \mathbf{x}^\top W \in \mathbb{R}^{C}, \qquad s_c(\mathbf{x}) = \mathbf{x}^\top \mathbf{w}_c.
$$

כלל החיזוי (עבור שני המודלים שלנו) הוא:

$$
\hat{y}(\mathbf{x}) = \arg\max_{c \in \\{0,\ldots,C-1\\}} s_c(\mathbf{x}).
$$

**Bias trick.** במקום לשמור וקטור הטיה $\mathbf{b}$ נפרד, אנחנו מוסיפים לכל דגימה עמודה קבועה של 1, כך ש־$D' = D+1$. העמודה האחרונה של $W$ ממלאת את תפקיד ה־bias. זה מפשט את המימוש לכפל מטריצות יחיד.

### פרספטרון רב-מחלקתי

**Loss (margin-based multiclass perceptron).** במימוש שלנו, ההפסד שמחזיר `perceptron_loss_naive` הוא הפסד מרג'יני. עבור דגימה $i$:

$$
\hat{y}_i = \arg\max_c \mathbf{x}_i^\top \mathbf{w}_c,
\qquad
L_i = \max\left(0, s_{\hat{y}_i}(\mathbf{x}_i) - s_{y_i}(\mathbf{x}_i)\right).
$$

במימוש לפי הבהרת המרצה, אם הדגימה מסווגת נכון אז $\hat{y}_i=y_i$ ולכן $L_i=0$. אם היא מסווגת לא נכון, ההפסד הוא הפער שבו המחלקה שנבחרה ניצחה את המחלקה הנכונה. אין כאן margin קבוע של $+1$.

ההפסד הממוצע בבאץ':

$$
L_{\text{perc}}(W) = \frac{1}{N}\sum_{i=1}^{N} L_i.
$$

**Update rule.** הגרדיאנט שנבנה בקוד ממש את כלל העדכון הקלאסי של פרספטרון רב-מחלקתי: עבור דגימה שסווגה שגוי ($\hat{y}_i \neq y_i$) נגדיר

$$
\frac{\partial L_i}{\partial \mathbf{w}_{y_i}} = -\mathbf{x}_i, \qquad
\frac{\partial L_i}{\partial \mathbf{w}_{\hat{y}_i}} = +\mathbf{x}_i,
$$

ועבור דגימה שסווגה נכון הגרדיאנט הוא אפס. לאחר צעד SGD סטנדרטי $W \leftarrow W - \eta \nabla_W L$, זה בדיוק מוסיף $\eta \mathbf{x}_i$ לעמודת המחלקה הנכונה ומוריד $\eta \mathbf{x}_i$ מעמודת המחלקה השגויה – כלומר **מגביר את הציון של המחלקה הנכונה ומחליש את הציון של השגויה**.

> הערה חשובה: $L_{\text{perc}}$ אינו גזיר חלק – אנחנו משתמשים ב־**sub-gradient**. זו הסיבה שהגרדיאנט לא עשיר במידע: הוא או אפס (סיווג נכון) או $\pm \mathbf{x}_i$.

### Softmax + Cross-Entropy

כאן עוברים ממודל "מבוסס טעויות" למודל הסתברותי.

**Softmax.** נגדיר התפלגות הסתברות מעל המחלקות:

$$
p(y=c \mid \mathbf{x}; W) = \frac{\exp(s_c(\mathbf{x}))}{\sum_{k=0}^{C-1} \exp(s_k(\mathbf{x}))}, \qquad c=0,\ldots,C-1.
$$

**Numerical stability.** מימושית, $\exp$ על ערכים גדולים גולש. טריק סטנדרטי:

$$
\frac{\exp(s_c)}{\sum_k \exp(s_k)} = \frac{\exp(s_c - m)}{\sum_k \exp(s_k - m)}, \qquad m = \max_k s_k.
$$

זו בדיוק השורה `scores -= np.max(scores, axis=1, keepdims=True)` בקוד.

**Cross-Entropy loss (NLL).** עבור labels $y_i$, ההפסד הממוצע הוא:

$$
L_{\text{CE}}(W) = -\frac{1}{N} \sum_{i=1}^{N} \log p(y=y_i \mid \mathbf{x}_i; W).
$$

**Gradient.** תוצאה יפהפיה שמוכחת בהרצאה היא שהגרדיאנט ביחס לציונים $\mathbf{s}_i$ הוא:

$$
\frac{\partial L_i}{\partial \mathbf{s}_i} = \mathbf{p}_i - \mathbf{e}_{y_i},
$$

כאשר $\mathbf{e}_{y_i}$ הוא וקטור one-hot של התווית האמיתית. בעזרת כלל השרשרת ($\mathbf{s}_i = \mathbf{x}_i^\top W$):

$$
\nabla_W L_{\text{CE}} \\;=\\; \frac{1}{N} X^\top (P - Y_{\text{onehot}})
$$

שבקוד זה בדיוק:
```python
dscores = probs.copy()
dscores[np.arange(N), y] -= 1.0
dW = X.T @ dscores / N
```

### Stochastic Gradient Descent

במקום לחשב גרדיאנט על כל $N$ הדגימות, בכל איטרציה דוגמים mini-batch $\mathcal{B}\subseteq\mathcal{D}$, $|\mathcal{B}|=b$, ומעדכנים:

$$
W \leftarrow W - \eta \cdot \nabla_W L_{\mathcal{B}}(W).
$$

יתרונות: סיבוכיות חישובית נמוכה בכל צעד, רעש שמאפשר יציאה ממינימום לוקלי. חסרונות: הגרדיאנט רועש, ולכן דרוש `learning_rate` קטן מספיק. בקוד מדגמים **with replacement** דרך `np.random.choice` – שיטה מהירה ומקובלת ל־SGD.

---

## זרימת הקוד

1. **טעינת נתונים** (במחברת): הורדת CIFAR-10 מהאוניברסיטת טורונטו, קריאת 5 batches של 10,000 תמונות כל אחד.
2. **בחירת 3 מחלקות** (במחברת): מסננים את הסט ל־3 מתוך 10 מחלקות.
3. **עיבוד מקדים** (במחברת):
   - Flatten: $32\times32\times3 \to 3072$.
   - חיסור ממוצע התמונות ($\bar{\mathbf{x}} = \frac{1}{N}\sum \mathbf{x}_i$) – מרכז את הנתונים סביב 0.
   - הוספת עמודת 1 (**bias trick**).
4. **פיצול**: train / val / test.
5. **אימון**: `model.train(...)` רץ SGD ומחזיר `loss_history`.
6. **הערכה**: `model.calc_accuracy(...)` על val / test.
7. **סריקת היפר-פרמטרים**: `tune_perceptron` בוחר את הקומבינציה הטובה ביותר לפי val accuracy.

---

## פירוט כל פונקציה בקוד

להלן הסבר פונקציה-פונקציה מתוך `linear_models.py`, בסדר שבו הן מופיעות בקובץ. לכל פונקציה מוצגים: **מה היא עושה**, **איך היא עושה זאת** ו־**הקישור לתאוריה**.

### `class LinearClassifier`

מחלקת בסיס מופשטת שבה יושבים החלקים שאינם תלויים באלגוריתם הספציפי: אתחול משקלות, חיזוי גנרי, חישוב דיוק, ולולאת אימון SGD. המחלקות `LinearPerceptron` ו־`LogisticRegression` יורשות ממנה ורק **דורסות את `loss`** (ולעיתים את `predict`).

#### `__init__(self, X, y)`

```23:43:linear_models.py
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        ...
        N, D = X.shape
        C = int(np.max(y)) + 1
        # Initialize weights with small random values
        self.W = 0.001 * np.random.randn(D, C)
        self.num_classes = C
        self.num_features = D
```

**מה**: יוצרת מטריצת משקלות $W \in \mathbb{R}^{D\times C}$ באתחול אקראי קטן מ־$\mathcal{N}(0, 0.001^2)$.

**למה אתחול קטן**: אתחול אקראי שובר סימטריה בין המחלקות (אם $W=0$ כל המחלקות מקבלות אותו ציון ו־softmax מחזיר התפלגות אחידה; יש גרדיאנט, אבל כל המחלקות "זזות יחד"). הסטייה הקטנה ($10^{-3}$) מבטיחה שהציונים ההתחלתיים קטנים מאוד – כך שה־softmax קרוב להתפלגות אחידה וה־CE ההתחלתי קרוב ל־$\log C$.

**קישור לתאוריה**: מתאים לעיקרון של *symmetry breaking initialization* שנלמד בהקשר של רשתות עמוקות; עבור מודל לינארי הוא פחות קריטי, אך עדיין חשוב.

#### `predict(self, X)`

ב־`LinearClassifier` זו מתודה מופשטת שזורקת `NotImplementedError`. כל מחלקה יורשת חייבת לממש אותה.

#### `calc_accuracy(self, X, y)`

```63:97:linear_models.py
    def calc_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        ...
        y_pred = self.predict(X)
        accuracy = float(np.mean(y_pred == y))
        return accuracy
```

**מה**: מחזירה את $\operatorname{acc} = \frac{1}{M}\sum_{i=1}^{M} \mathbb{1}[\hat{y}_i = y_i]$ – שיעור הדגימות שסווגו נכון.

**איך**: משתמש ב־`self.predict` (פולימורפיזם – יקרא ל־`predict` הספציפי של התת־מחלקה) ואז ממוצע של שוויונים בוליאניים.

**קישור לתאוריה**: זו מטריקת הערכה **שאינה גזירה** – לכן לא מבצעים עליה אופטימיזציה ישירה, אלא משתמשים בה רק להערכה (val/test). בפועל, הפסד פרוקסי (0/1 עבור פרספטרון, CE עבור Softmax) משמש לאימון.

#### `train(self, X, y, learning_rate, num_iters, batch_size, verbose)`

```99:200:linear_models.py
    def train(self, X, y, learning_rate=1e-3, num_iters=100, batch_size=200, verbose=False):
        ...
        for i in range(num_iters):
            batch_idx = np.random.choice(num_instances, batch_size, replace=True)
            X_batch = X[batch_idx]; y_batch = y[batch_idx]
            loss, grad = self.loss(X_batch, y_batch)
            loss_history.append(loss)
            self.W -= learning_rate * grad
        return loss_history
```

**מה**: לולאת SGD קלאסית. בכל איטרציה:

1. **דגימת mini-batch** בגודל `batch_size` **with replacement** (פשוט ומהיר; ראו דיון על הבחירה בסעיף SGD למעלה).
2. **חישוב loss וגרדיאנט** דרך `self.loss(...)` – שים לב ש־`self.loss` עובר דריסה בכל תת־מחלקה, כך שהמתודה ה*גנרית* הזו עובדת עבור פרספטרון ועבור logistic regression בדיוק אותו דבר.
3. **שמירת ה־loss** ב־`loss_history` – לשרטוט עקומת האימון בהמשך.
4. **עדכון משקלות** בצעד vanilla gradient descent: $W \leftarrow W - \eta \cdot \nabla_W L$.

**קישור לתאוריה**: הלולאה הזו היא **המימוש המדויק** של הצורה החישובית של SGD שנלמדה בהרצאה. הבחירה להשתמש ב־`loss_history` מאפשרת לבחון **התכנסות**: אם ה־loss לא יורד – `learning_rate` גדול מדי או קטן מדי, אם הוא יורד וחוזר לטפס – אולי צעד לא יציב.

#### `loss(self, X, y)`

מחלקת בסיס; זורקת `NotImplementedError`. נדרס ב־`LinearPerceptron` ו־`LogisticRegression`.

---

### `class LinearPerceptron(LinearClassifier)`

מימוש של סיווגן פרספטרון רב-מחלקתי.

#### `__init__`

```226:249:linear_models.py
    def __init__(self, X, y):
        super().__init__(X, y)
```

קריאה פשוטה לאב – אין כאן פרמטרים נוספים.

#### `predict(self, X)`

```252:285:linear_models.py
    def predict(self, X):
        scores = X @ self.W
        y_pred = np.argmax(scores, axis=1)
        return y_pred
```

**מה**: מבצע $\hat{y} = \arg\max_c \mathbf{x}^\top \mathbf{w}_c$.

**קישור לתאוריה**: זו בדיוק נוסחת החיזוי של classifier לינארי – המחלקה עם הציון הגבוה ביותר.

#### `loss(self, X_batch, y_batch)`

```287:306:linear_models.py
    def loss(self, X_batch, y_batch):
        return perceptron_loss_naive(self.W, X_batch, y_batch)
```

**מה**: מחזיר את הפסד הפרספטרון הרב-מחלקתי ואת הגרדיאנט שלו. לפי הבהרת פיאצה, זו הקריאה הנכונה במחלקה `LinearPerceptron`.

---

### `class LogisticRegression(LinearClassifier)`

מסווגן רב-מחלקתי באמצעות Softmax + Cross-Entropy.

#### `__init__`

```316:331:linear_models.py
    def __init__(self, X, y):
        self.W = None
        super().__init__(X, y)
```

שוב, שימוש ב־`super().__init__` – אתחול רגיל.

#### `predict(self, X)`

```333:366:linear_models.py
    def predict(self, X):
        scores = X @ self.W
        probs = softmax(scores)
        y_pred = np.argmax(probs, axis=1)
        return y_pred
```

**מה**: מחשב הסתברויות softmax ובוחר את המחלקה בעלת ההסתברות הגבוהה.

**נקודה מתמטית יפה**: מכיוון ש־$\operatorname{softmax}$ היא פונקציה מונוטונית (עולה ביחס לציון של אותה מחלקה), מתקיים:

$$
\arg\max_c \\; p_c(\mathbf{x}) = \arg\max_c \\; s_c(\mathbf{x}).
$$

לכן הקריאה ל־`softmax` **לא משנה את התחזית** – היא נשארת כאן ליישור קו עם הפרק במחברת ("Softmax section") ולצורכי ניפוי שגיאות (אפשר להציג probabilities).

#### `loss(self, X_batch, y_batch)`

```368:387:linear_models.py
    def loss(self, X_batch, y_batch):
        return softmax_cross_entropy_vectorized(self.W, X_batch, y_batch)
```

משתמש במימוש הווקטוריאלי (מהיר). נצלול אליו מייד.

---

### `perceptron_loss_naive(W, X, y)`

```389:463:linear_models.py
def perceptron_loss_naive(W, X, y):
    ...
    for i in range(N):
        scores_i = X[i] @ W
        pred = int(np.argmax(scores_i))
        true_label = int(y[i])
        if pred != true_label:
            loss += scores_i[pred] - scores_i[true_label]
            dW[:, true_label] -= X[i]
            dW[:, pred]       += X[i]
    loss /= N
    dW /= N
    return loss, dW
```

**מה**: מחשב הפסד פרספטרון וגרדיאנט בעבור batch, עם לולאה מפורשת על הדגימות.

**איך בדיוק**:
- עבור כל דגימה מחשב $\mathbf{s}_i = \mathbf{x}_i^\top W$ ו־$\hat{y}_i = \arg\max_c s_{i,c}$.
- אם סווג שגוי (ורק אז):
  - **Loss**: מוסיף את המרג'ין $s_{\hat{y}_i} - s_{y_i}$.
  - **Gradient**: $\mathbf{dW}[:, y_i] \mathrel{-}= \mathbf{x}_i$ ו־$\mathbf{dW}[:, \hat{y}_i] \mathrel{+}= \mathbf{x}_i$.
- בסוף מחלק ב־$N$: ההפסד הוא **margin loss ממוצע**, והגרדיאנט מנורמל כך ש־`learning_rate` לא תלוי בגודל ה־batch.

**למה**: זה מימוש ישיר של **sub-gradient** של הפסד הפרספטרון המרג'יני עם ה־update rule הקלאסי של פרספטרון:

$$
\mathbf{w}_{y_i} \leftarrow \mathbf{w}_{y_i} + \eta \mathbf{x}_i, \qquad \mathbf{w}_{\hat{y}_i} \leftarrow \mathbf{w}_{\hat{y}_i} - \eta \mathbf{x}_i \qquad \text{(רק אם טעות)}.
$$

שים לב שאחרי הצעד $W \leftarrow W - \eta \nabla_W L$ הסימנים מתהפכים, ובדיוק מתקבלת הנוסחה למעלה.

**קישור לתאוריה**: זוהי הצורה הרב-מחלקתית של Rosenblatt Perceptron Learning Rule, המוצגת בהרצאה. זו הגרסה ה"איטית" (loops ב־Python) שמוגשת למטרת הבנה – ולאימות מול הגרסה הווקטוריאלית במבחני gradient check במחברת.

---

### `softmax_cross_entropy(W, X, y)`

```465:555:linear_models.py
def softmax_cross_entropy(W, X, y):
    ...
    probs_all = np.zeros((N, C))
    for i in range(N):
        scores_i = X[i] @ W
        scores_i -= np.max(scores_i)          # stability
        exp_scores = np.exp(scores_i)
        probs_all[i] = exp_scores / np.sum(exp_scores)

    correct_probs = probs_all[np.arange(N), y]
    loss = float(np.mean(-np.log(correct_probs + 1e-12)))

    dscores = probs_all.copy()
    dscores[np.arange(N), y] -= 1.0
    dW = X.T @ dscores / N
    return loss, dW
```

**מה**: גרסת ה**ייחוס** של softmax + cross-entropy, עם לולאה מפורשת ל־forward pass של כל דגימה.

**מבנה הפונקציה בשלושה שלבים**:

1. **Forward pass (softmax):** עבור כל דגימה $i$, מחשב $\mathbf{s}_i$, מחסר `max` (stability), מעלה ב־$\exp$, ומנרמל.
2. **Loss:** בוחר את ההסתברות של המחלקה הנכונה לכל דגימה (`probs_all[np.arange(N), y]` – **fancy indexing**), לוקח $-\log$ וממוצע. ה־`+ 1e-12` מונע $\log(0)$ כש־probability קרובה ל־0 לחלוטין.
3. **Backward pass:** הגרדיאנט ביחס לציונים הוא $P - Y_{\text{onehot}}$, ולכן `dscores = probs - onehot(y)`; ואז `dW = X.T @ dscores / N` – כלל השרשרת.

**קישור לתאוריה**: ראו סעיף Softmax + CE ב**רקע התיאורטי**. זו הגרסה שמממשת *בדיוק* את הנוסחאות, ב־ O(N·D·C) אבל עם לולאה – לכן איטית בפייתון.

---

### `softmax(x)`

```559:590:linear_models.py
def softmax(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
    return probs
```

**מה**: softmax שורה־שורה, ווקטוריאלית לגמרי.

**איך**:
- `np.max(..., axis=1, keepdims=True)` – המקסימום בכל שורה, שמירה על $N\times 1$ כך שייעשה broadcast.
- חיסור המקסימום = **log-sum-exp trick** לייצוב נומרי.
- חלוקה עם `keepdims=True` מבטיחה שאין ניסיון להסתכלות אגרסיבית על ממדים.

**קישור לתאוריה**: מימוש ישיר של $\operatorname{softmax}(\mathbf{s})_c = \frac{e^{s_c}}{\sum_k e^{s_k}}$ בגרסה נומרית יציבה.

---

### `softmax_cross_entropy_vectorized(W, X, y)`

```592:685:linear_models.py
def softmax_cross_entropy_vectorized(W, X, y):
    ...
    scores = X @ W                                      # (N, C)
    scores -= np.max(scores, axis=1, keepdims=True)     # stability
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    correct_probs = probs[np.arange(N), y]
    loss = float(np.mean(-np.log(correct_probs + 1e-12)))

    dscores = probs.copy()
    dscores[np.arange(N), y] -= 1.0
    dW = X.T @ dscores / N
    return loss, dW
```

**מה**: אותו החישוב כמו `softmax_cross_entropy`, אבל **ללא לולאות ב־Python** – הכל matmul-ים ופעולות numpy קולקטיביות.

**השוואה מול הגרסה ה־naive**:
| שלב | Naive | Vectorized |
|---|---|---|
| Forward | לולאה של $N$ כפלי וקטור-מטריצה | matmul יחיד `X @ W` |
| Softmax | לכל דגימה בנפרד | `np.max(..., axis=1)` ופעולות broadcast |
| Gradient | לולאה + צבירה ל־`dW` | `X.T @ dscores` יחיד |

**קישור לתאוריה**: זו אותה מתמטיקה בדיוק (ובמחברת מאמתים את זה מספרית עם gradient check), רק שמבחינת יעילות numpy זה גדל של סדרי גודל – ולכן זו הגרסה שמשמשת את `LogisticRegression.loss` באימון.

---

### `tune_perceptron(ModelClass, X_train, y_train, X_val, y_val, learning_rates, batch_sizes, num_iters, ...)`

```688:797:linear_models.py
def tune_perceptron(ModelClass, X_train, y_train, X_val, y_val,
                    learning_rates, batch_sizes, *, num_iters=500, ...):
    ...
    for lr in learning_rates:
        for batch_size in batch_sizes:
            model = ModelClass(X_train, y_train, **model_kwargs)
            model.train(X_train, y_train,
                        learning_rate=lr, num_iters=num_iters,
                        batch_size=batch_size, verbose=False)
            train_acc = model.calc_accuracy(X_train, y_train)
            val_acc   = model.calc_accuracy(X_val, y_val)
            results[(lr, batch_size)] = (train_acc, val_acc)
            if val_acc > best_val:
                best_val = val_acc
                best_model = model
    return results, best_model, best_val
```

**מה**: **Grid search** פשוט מעל שני היפר-פרמטרים: `learning_rate` ו־`batch_size`. מאמן מודל חדש לכל קומבינציה ומחזיר את הטוב ביותר לפי val accuracy.

**למה חשוב**:
- `learning_rate` קובע יציבות/התכנסות של SGD.
- `batch_size` קובע את רעש הגרדיאנט: batch גדול => גרדיאנט מדויק אבל איטר' יקרה; batch קטן => רעש גבוה, יותר סיכוי לצאת ממינימום לוקלי.
- חשוב ליצור **מודל חדש** לכל קומבינציה (`ModelClass(X_train, y_train)`) כדי שלא נשתמש במשקלות שכבר עברו אימון בקומבינציה הקודמת.
- הבחירה מתבצעת לפי **val**, **לא** test, כדי לשמור על הערכה הוגנת של generalization.

**קישור לתאוריה**: זוהי הפרדיגמה של **model selection** – מפצלים את הדאטא ל־train/val/test כדי לבחור היפר-פרמטרים ללא זיהום של סט הבדיקה.

---

## היפר-פרמטרים וסריקה

הערכים שנסרקים במחברת (ראו תא `tune_perceptron(...)` ו־`tune_perceptron(LogisticRegression, ...)`):
- `learning_rates` – טיפוסית $\\{10^{-7}, 10^{-6}, \ldots, 10^{-3}\\}$. ערכים גדולים מדי => loss מתפוצץ; ערכים קטנים מדי => אימון איטי.
- `batch_sizes` – טיפוסית $\\{64, 128, 256\\}$ או בסדר גודל דומה.
- `num_iters` – מספר צעדי SGD; מספיק גדול כדי שה־loss יתיישר.

בסוף, התוצאה של הסריקה היא:
- `results[(lr, bs)] = (train_acc, val_acc)` – למילוי טבלה.
- `best_model` – לאימות סופי על test.
- `best_val` – ה־val accuracy הטובה ביותר.

---

## הוראות הרצה

המחברת מיועדת ל־**Google Colab** (שם יש הרכבת Drive ב־`/content/drive`).

1. להעלות את `linear_models.py` ו־`ex1.ipynb` לתיקייה ב־Drive (בברירת מחדל: `/content/drive/MyDrive/HW1`).
2. להריץ את התא הראשון במחברת – הוא מרכיב את Drive ומוסיף את התיקייה ל־`sys.path` כדי שאפשר יהיה לעשות `import linear_models`.
3. להריץ את התאים לפי הסדר. CIFAR-10 יורד אוטומטית ומחולץ ל־`datasets/cifar10/`.

**להרצה מקומית** (ללא Colab):
1. להתקין ספריות: `pip install numpy matplotlib jupyter`.
2. להעיר את שתי שורות `from google.colab import drive` / `drive.mount(...)` / `sys.path.append(...)` ולוודא ש־`linear_models.py` בתיקייה הנוכחית.
3. להריץ `jupyter notebook ex1.ipynb`.

---

## סיכום הקשר קוד–תאוריה

| מושג תיאורטי | איפה בקוד |
|---|---|
| מטריצת משקלות $W \in \mathbb{R}^{D\times C}$ | `LinearClassifier.__init__` |
| חיזוי $\arg\max_c \mathbf{x}^\top \mathbf{w}_c$ | `LinearPerceptron.predict`, `LogisticRegression.predict` |
| Bias trick (עמודת 1) | preprocessing במחברת |
| פרספטרון 0/1 loss + sub-gradient | `perceptron_loss_naive` |
| Softmax יציב נומרית | `softmax`, ובתוך `softmax_cross_entropy*` |
| Cross-entropy loss | הנוסחה `-mean(log correct_probs)` בשתי פונקציות ה־CE |
| גרדיאנט $\nabla_W L = X^\top(P - Y_{\text{onehot}})/N$ | `dscores = probs.copy(); dscores[..., y] -= 1; dW = X.T @ dscores / N` |
| SGD step $W \leftarrow W - \eta \nabla L$ | `self.W -= learning_rate * grad` בתוך `LinearClassifier.train` |
| Mini-batch sampling | `np.random.choice(..., replace=True)` בתוך `train` |
| Model selection | `tune_perceptron` |

בכך אנו מקיימים מיפוי **אחד-לאחד** בין הנוסחאות בהרצאה לבין השורות המדויקות בקוד.
