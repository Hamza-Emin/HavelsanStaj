import re
from fuzzywuzzy import process
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk
import pandas 
import string


dframe = pandas.read_csv("vodafone_data.csv", encoding='utf-8', on_bad_lines='skip', sep=';')
print(dframe)


def clean_text(text):
    text = " ".join(str(text).split())
    text = text.lower()
    text = text.replace("\\n", " ")
    text = re.sub("[0-9]+", "", text)
    text = re.sub("%|(|)|-", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return "".join(text)


dframe["Clean"] = dframe.apply(lambda row: clean_text(row["Explanation"]), axis=1)

nltk.download('stopwords')


aspect_dict = {
    "Vodafone": ["vodafone"],
    "Türkcell": ["türkcell", "turkcell"],
    "Türk Telekom": ["türk telekom", "türk teleko", "telekom"],
    "Fatura": ["fatura", "ödeme", "borç", "ücret"],
    "Şebeke": ["şebeke", "ağ", "bağlantı", "kapsama"],
    "İnternet": ["internet", "data", "veri", "web"],
    "Müşteri Hizmetleri": ["müşteri hizmetleri", "destek", "çağrı merkezi"],
    "Fiyat": ["fiyat", "ücret", "maliyet", "tarife", "tutar"],
    "Çekim Gücü": ["çekim gücü", "sinir gücü", "sinir kalitesi", "çubuklar"],
    "Hız": ["hız", "yavaş", "gecikme", "ping", "download"],
    "Altyapı": ["altyapı", "kablo", "fiber", "donanım"],
    "Kapsama Alanı": ["kapsama alanı", "kapsama", "bölge", "kırsal", "şehir"],
    "Hizmet Kalitesi": ["hizmet kalitesi", "kalite", "hizmet", "tatmin"],
    "Fatura Kesme": ["fatura kesme", "ödeme iptali", "servis kesme", "hizmet durdurma"],
    "Pandemi": ["pandemi", "covid", "corona", "karantina", "salgın"],
    "Kampanya": ["kampanya", "promosyon", "indirim", "fırsat"],
    "Mobil İnternet": ["mobil internet", "mobil data", "cep interneti", "gsm internet"],
    "4.5G/3G": ["4.5g", "3g", "5g"],
    "Modem": ["modem", "router", "cihaz", "bağlantı cihazı"],
    "Güvenilirlik": ["güvenilirlik", "kararlılık", "kesinti", "sürekli bağlantı"],
    "Dijital Operatör": ["dijital operatör", "online işlemler", "mobil uygulama", "internet sitesi"],
    "Ödeme Kolaylığı": ["ödeme kolaylığı", "kredi kartı", "online ödeme", "ödeme seçenekleri"]
}

def label_aspect(text):
    aspects = []
    text = text.lower()  

  
    def find_closest_aspect(word):
        aspect_name, score = process.extractOne(word, [item for sublist in aspect_dict.values() for item in sublist])
        if score >= 65:  
            for aspect, variations in aspect_dict.items():
                if aspect_name in variations:
                    return aspect
        return None

 
    for aspect, variations in aspect_dict.items():
        for variation in variations:
            if re.search(r'\b' + re.escape(variation) + r'\b', text):
                aspects.append(aspect)
                break


    if not any(operator in aspects for operator in ["Vodafone", "Türkcell", "Türk Telekom"]):
        aspects.append("Vodafone")

    return aspects

dframe["Aspects"] = dframe["Clean"].apply(label_aspect)


print(dframe["Aspects"].to_list())


dframe.to_csv("output.csv", sep=';', encoding='utf-8')
