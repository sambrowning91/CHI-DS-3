from nltk.stem.wordnet import WordNetLemmatizer
import re
import string
import nltk

#from nltk.stem.snowball import SnowballStemmer
#Pstemmer = PorterStemmer()
#Sstemmer = SnowballStemmer("english")
#[Sstemmer.stem(plural) for plural in plurals]
from nltk.corpus import stopwords
def process(text,stopwords=True):

	Wstemmer = WordNetLemmatizer()
	
	#remove extra spaces
	text = re.sub("\s+"," ",text)

	#remove non-printable characters
	text = re.sub("[^\x00-\x7f]","",text)

	#remove line breaks and tabs
	text = re.sub("\r|\n|\t"," ",text)

	#remove punctuation
#	text = text.translate(None, string.punctuation)
	sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
	sentences = [sentence.translate(None, string.punctuation) for sentence in sent_detector.tokenize(text)]
	text = " ".join(sentences)

	#remove digits
	text = text.translate(None, string.digits)

	#convert text to all lowercase
	text = text.lower().split(" ")
	text = " ".join([Wstemmer.lemmatize(word) for word in text])
	text =  ''.join(filter(lambda x:x in string.printable, text)) 
	#remove stopwords
	if stopwords == True:
		stopwords=["a","about","above","across","after","again","against","all","almost","alone","along","already","also","although","always","among","an","and","another","any","anybody","anyone","anything","anywhere","are","area","areas","around","as","ask","asked","asking","asks","at","away","b","back","backed","backing","backs","be","became","because","become","becomes","been","before","began","behind","being","beings","best","better","between","big","both","but","by","c","came","can","cannot","case","cases","certain","certainly","clear","clearly","come","could","d","did","differ","different","differently","do","does","done","down","down","downed","downing","downs","during","e","each","early","either","end","ended","ending","ends","enough","even","evenly","ever","every","everybody","everyone","everything","everywhere","f","face","faces","fact","facts","far","felt","few","find","finds","first","for","four","from","full","fully","further","furthered","furthering","furthers","g","gave","general","generally","get","gets","give","given","gives","go","going","good","goods","got","great","greater","greatest","group","grouped","grouping","groups","h","had","has","have","having","he","her","here","herself","high","high","high","higher","highest","him","himself","his","how","however","i","if","important","in","interest","interested","interesting","interests","into","is","it","its","itself","j","just","k","keep","keeps","kind","knew","know","known","knows","l","large","largely","last","later","latest","least","less","let","lets","like","likely","long","longer","longest","m","made","make","making","man","many","may","me","member","members","men","might","more","most","mostly","mr","mrs","much","must","my","myself","n","necessary","need","needed","needing","needs","never","new","new","newer","newest","next","no","nobody","non","noone","not","nothing","now","nowhere","number","numbers","o","of","off","often","old","older","oldest","on","once","one","only","open","opened","opening","opens","or","order","ordered","ordering","orders","other","others","our","out","over","p","part","parted","parting","parts","per","perhaps","place","places","point","pointed","pointing","points","possible","present","presented","presenting","presents","problem","problems","put","puts","q","quite","r","rather","really","right","right","room","rooms","s","said","same","saw","say","says","second","seconds","see","seem","seemed","seeming","seems","sees","several","shall","she","should","show","showed","showing","shows","side","sides","since","small","smaller","smallest","so","some","somebody","someone","something","somewhere","state","states","still","still","such","sure","t","take","taken","than","that","the","their","them","then","there","therefore","these","they","thing","things","think","thinks","this","those","though","thought","thoughts","three","through","thus","to","today","together","too","took","toward","turn","turned","turning","turns","two","u","under","until","up","upon","us","use","used","uses","v","very","w","want","wanted","wanting","wants","was","way","ways","we","well","wells","went","were","what","when","where","whether","which","while","who","whole","whose","why","will","with","within","without","work","worked","working","works","would","x","y","year","years","yet","you","young","younger","youngest","your","yours","z"]
		#text= " ".join(filter(lambda x: x not in stopwords, text.split(" ")))
		text = re.sub("\\b"+"\\b|\\b".join(stopwords)+"\\b","",text)
	text = re.sub("\s+"," ",text)
	return text