import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


dict_emotions = {

    1: 'Adaptacion de los acogidos',  
    2: 'Les gustaria contactar con otra familia acogedora',   
    3: 'Quiere saber si habria problema a acoger siendo soltero',
    4: 'Apoyo una vez que ya he acogido',
    5: 'Aqui tienes las Ayudas economicas disponibles',
    6: 'Solucion de miedo a hacerlo mal con el acogido',
    7: 'Lo que mas te preocupa son los vinculos generados con el acogido'

}



    
def signs_texts(text):
    signos = re.compile("(\.)|(\;)|(\:)|(\!)|(\?)|(\¿)|(\@)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
    return signos.sub(' ', text.lower())
    
def remove_stopwords(df):
    spanish_stopwords = stopwords.words('spanish')
    return " ".join([word for word in df.split() if word not in spanish_stopwords])
    
def spanish_stemmer(x):
    stemmer = SnowballStemmer('spanish')
    return " ".join([stemmer.stem(word) for word in x.split()])