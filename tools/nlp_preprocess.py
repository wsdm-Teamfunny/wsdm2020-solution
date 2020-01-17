import re
import time
import  numpy as np
import nltk
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

def tokenize(sentence):
    '''
        去除多余空白、分词、词性标注
    '''
    sentence = re.sub(r'\s+', ' ', sentence)
    token_words = word_tokenize(sentence)  # 输入的是列表
    token_words = pos_tag(token_words)
    return token_words

def stem(token_words):
    '''
        词形归一化
    '''
    wordnet_lematizer = WordNetLemmatizer()  # 单词转换原型
    words_lematizer = []
    for word, tag in token_words:
        if tag.startswith('NN'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='n')  # n代表名词
        elif tag.startswith('VB'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='v')  # v代表动词
        elif tag.startswith('JJ'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='a')  # a代表形容词
        elif tag.startswith('R'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='r')  # r代表代词
        else:
            word_lematizer = wordnet_lematizer.lemmatize(word)
        words_lematizer.append(word_lematizer)
    return words_lematizer


sr = stopwords.words('english')


def delete_stopwords(token_words):
    '''
        去停用词
    '''
    cleaned_words = [word for word in token_words if word not in sr]
    return cleaned_words


def is_number(s):
    '''
        判断字符串是否为数字
    '''
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


characters = [' ', ',', '.', 'DBSCAN', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '-', '...',
              '^', '{', '}']


def delete_characters(token_words):
    '''
        去除特殊字符、数字
    '''
    words_list = [word for word in token_words if word not in characters and not is_number(word)]
    return words_list


def to_lower(token_words):
    '''
        统一为小写
    '''
    words_lists = [x.lower() for x in token_words]
    return words_lists

def replace_process(line):
    m = replace = {
        'α': 'alpha',
        'β': 'beta',
        'γ': 'gamma',
        'δ': 'delta',
        'ε': 'epsilon',
        'ζ': 'zeta',
        'η': 'eta',
        'θ': 'theta',
        'ι': 'iota',
        'κ': 'kappa',
        'λ': 'lambda',
        'μ': 'mu',
        'ν': 'nu',
        'ξ': 'xi',
        'ο': 'omicron',
        'π': 'pi',
        'ρ': 'rho',
        'ς': 'sigma',
        'σ': 'sigma',
        'τ': 'tau',
        'υ': 'upsilon',
        'φ': 'phi',
        'χ': 'chi',
        'ψ': 'psi',
        'ω': 'omega',
        'ϑ': 'theta',
        'ϒ': 'gamma',
        'ϕ': 'phi',
        'ϱ': 'rho',
        'ϵ': 'epsilon',
        '𝛼': 'alpha',
        '𝛽': 'beta',
        '𝜀': 'epsilon',
        '𝜃': 'theta',
        '𝜏': 'tau',
        '𝜖': 'epsilon',
        '𝜷': 'beta',
    }
    empty_str = ['etc.','et al.','fig.','figure.','e.g.','(', ')','[', ']',';',':','!',',','.','?','"','\'', \
                 '%','>','<','+','&']
    m.update({s: ' ' for s in empty_str})
    
    for k, v in m.items():
        line = line.replace(k, v)
    line = ' '.join([s.strip() for s in line.split(' ') if s != ''])
    return line

def preprocess(line):
    '''
        文本预处理
    '''
    line = line.lower()
    line = replace_process(line)
    token_words = tokenize(line)
    token_words = stem(token_words)
    token_words = delete_stopwords(token_words)
    token_words = delete_characters(token_words)
    token_words = to_lower(token_words)
    return ' '.join(token_words)

if __name__ == '__main__':
    text = 'This experiment was conducted to determine whether feeding meal and hulls derived from genetically modified soybeans to dairy cows affected production measures and sensory qualities of milk. The soybeans were genetically modified (Event DAS-444Ø6-6) to be resistant to multiple herbicides. Twenty-six Holstein cows (13/treatment) were fed a diet that contained meal and hulls derived from transgenic soybeans or a diet that contained meal and hulls from a nontransgenic near-isoline variety. Soybean products comprised approximately 21% of the diet dry matter, and diets were formulated to be nearly identical in crude protein, neutral detergent fiber, energy, and minerals and vitamins. The experimental design was a replicated 2×2 Latin square with a 28-d feeding period. Dry matter intake (21.3 vs. 21.4kg/d), milk yield (29.3 vs. 29.4kg/d), milk fat (3.70 vs. 3.68%), and milk protein (3.10 vs. 3.12%) did not differ between cows fed control or transgenic soybean products, respectively. Milk fatty acid profile was virtually identical between treatments. Somatic cell count was significantly lower for cows fed transgenic soybean products, but the difference was biologically trivial. Milk was collected from all cows in period 1 on d 0 (before treatment), 14, and 28 for sensory evaluation. On samples from all days (including d 0) judges could discriminate between treatments for perceived appearance of the milk. The presence of this difference at d 0 indicated that it was likely not a treatment effect but rather an initial bias in the cow population. No treatment differences were found for preference or acceptance of the milk. Overall, feeding soybean meal and hulls derived from this genetically modified soybean had essentially no effects on production or milk acceptance when fed to dairy cows. '
    text = 'Pyrvinium is a drug approved by the FDA and identified as a Wnt inhibitor by inhibiting Axin degradation and stabilizing 尾-catenin, which can increase Ki67+ cardiomyocytes in the peri-infarct area and alleviate cardiac remodeling in a mouse model of MI . UM206 is a peptide with a high homology to Wnt-3a/5a, and acts as an antagonist for Frizzled proteins to inhibit Wnt signaling pathway transduction. UM206 could reduce infarct size, increase the numbers of capillaries, decrease myofibroblasts in infarct area of post-MI heart, and ultimately suppress the development of heart failure . ICG-001, which specifically inhibits the interaction between 尾-catenin and CBP in the Wnt canonical signaling pathway, can promote the differentiation of epicardial progenitors, thereby contributing to myocardial regeneration and improving cardiac function in a rat model of MI . Small molecules invaliding Porcupine have been further studied, such as WNT-974, GNF-6231 and CGX-1321. WNT-974 decreases fibrosis in post-MI heart, with a mechanism of preventing collagen production in cardiomyocytes by blocking secretion of Wnt-3, a pro-fibrotic agonist, from cardiac fibroblasts and its signaling to cardiomyocytes . The phosphorylation of DVL protein is decreased in both the canonical and non-canonical Wnt signaling pathways by WNT-974 administration . GNF-6231 prevents adverse cardiac remodeling in a mouse model of MI by inhibiting the proliferation of interstitial cells, increasing the proliferation of Sca1+ cardiac progenitors and reducing the apoptosis of cardiomyocytes [[**##**]]. Similarly, we demonstrate that CGX-1321, which has also been applied in a phase I clinical trial to treat solid tumors ({"type":"clinical-trial","attrs":{"text":"NCT02675946","term_id":"NCT02675946"}}NCT02675946), inhibits both canonical and non-canonical Wnt signaling pathways in post-MI heart. CGX-1321 promotes cardiac function by reducing fibrosis and stimulating cardiomyocyte proliferation-mediated cardiac regeneration in a Hippo/YAP-independent manner . These reports implicate that Wnt pathway inhibitors are a class of potential drugs for treating MI through complex mechanisms, including reducing cardiomyocyte death, increasing angiogenesis, suppressing fibrosis and stimulating cardiac regeneration.'
    token_words = tokenize(text)
    print(token_words)
    token_words = stem(token_words)  # 单词原型
    token_words = delete_stopwords(token_words)  # 去停
    token_words = delete_characters(token_words)
    token_words = to_lower(token_words)
    print(token_words)
