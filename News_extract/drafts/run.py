
from News_Extraction.ltp_tool import sentence_embedding,LTP
from gensim.models import Word2Vec

ltp = LTP()
model = Word2Vec.load('./final_model')
sentence_embedding = sentence_embedding(model,'./words.txt',0.001)
result = open('./news_extraction.csv','w',encoding='utf-8')
line = '李连杰突然宣布，将推出娱乐圈，以后将把自己中心放在慈善活动，希望各位粉丝不要失望'
line = str(line.strip())
wordslist = ltp.cut(line)
keywords_index, postags, arcs = ltp.get_dependtree_root_index(wordslist)
print(keywords_index)
keyword = wordslist[keywords_index]
main_index = ltp.get_sbv_id(postags, arcs,keywords_index)
main = wordslist[main_index]
# content = ltp.find_content(wordslist,keywords_index)
sentences = ltp.keyword_senteces(wordslist,postags,keywords_index)
print('sentences:',sentences)
content = sentence_embedding.get_final_sents(sentences)
result.write(str(main) + '-->' + str(keyword) + '-->' + str(content))
print(str(main) + '-->' + str(keyword) + '-->' + str(content))























