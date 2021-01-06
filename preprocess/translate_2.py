import http.client
import hashlib
import json
import urllib
import random
import jieba
import os
# _train.py
import sys, os
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from utils import list_of_groups, throw_error
#from googletrans import Translator


# 调用百度翻译API将中文翻译成英文
def baidu_translate(content, fromLang, toLang):
    appid = '20201222000653224'
    secretKey = 'LRStwxmJfAHO99Rl9oqh'
    httpClient = None
    myurl = '/api/trans/vip/translate'
    q = content
    # fromLang = 'en'  # 源语言
    # toLang = 'zh'  # 翻译后的语言
    salt = random.randint(32768, 65536)
    sign = appid + q + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(
        q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
        salt) + '&sign=' + sign

    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)

        # response是HTTPResponse对象
        response = httpClient.getresponse()
        jsonResponse = response.read().decode("utf-8")  # 获得返回的结果，结果为json格式
        js = json.loads(jsonResponse)  # 将json格式的结果转换字典结构
        if "trans_result" in js:
            lens = len(js['trans_result'])
            samples = []
            for sam in js['trans_result']:
                samples.append(str(sam['dst']))
            return samples
        else:
            return None

        # dst = str(js["trans_result"][0]["dst"])  # 取得翻译后的文本结果
        # print(dst)  # 打印结果
        # return dst
    except Exception as e:
        #print('err:' + e)
        print(e)
    finally:
        if httpClient:
            httpClient.close()


# def google_translate(content, fromLang, toLang):
#     translator = Translator(service_urls=['translate.google.cn'])
#     text = translator.translate(content, src=fromLang, dest=toLang).text
#     return text

def preclear(src):
    # TODO 去除非中文字符
    constr = ""
    for ch in src:
        if ch >= u'\u4e00' and ch <= u'\u9fa5':
            if ch != " ":
                constr += ch
    # TODO 去除大量重复词语
    seq_new = []
    seq_list = jieba.cut(constr, cut_all=False)
    for token in seq_list:
        if token not in seq_new:
            seq_new.append(token)
    src = ''.join(seq_new)
    return src


def readAndWrite1(index1, fileReadPath, fileWritePath, srcLang, destLang):
    with open(fileReadPath, 'r', encoding='utf8') as fileread:
        with open(fileWritePath, 'a+', encoding='utf8') as filewrite:
            sum = 0
            for idx,line in enumerate(fileread.readlines()):
                if index1 != 1:
                    id, s1 = line.split('\t')
                    # line = preclear(s1)
                    line = s1
                else:
                    id, s1, s2 = line.split('\t')
                    # line = preclear(s1) + '\n' + preclear(s2)
                    line = s1 + '\n' + s2
                sum += 1
                while True:
                    text = baidu_translate(line, srcLang, destLang)
                    if text != None:
                        if index1 != 1:
                            filewrite.write(id + '\t' + text[0] + "\n")
                        else:
                            filewrite.write(id + '\t' + text[0] + '\t' + text[1] + '\n')
                        print("finish" + str(sum) + "translate task")
                        break



def main():
    root = "/share/home/crazicoco/competition/CPFC"
    filePathSrc = ['rawData/OCEMOTION_train_1.csv', 'rawData/OCNLI_train.csv', 'rawData/TNEWS_train.csv']
    filePathChinese = ['rawData/translation/chinese/OCEMOTION_train_1.csv', 'rawData/translation/chinese/OCNLI_train.csv', 'rawData/translation/chinese/TNEWS_train.csv']
    filePathEnglish = ['rawData/translation/english/OCEMOTION_train_1.csv', 'rawData/translation/english/OCNLI_train.csv', 'rawData/translation/english/TNEWS_train.csv']

    google_chinese = "zh-cn"
    baidu_chinese = "zh"
    english = "en"
    print("--------------------------start translate english------------------------")
    fileNum=len(filePathSrc)


    # TODO test chinese to english
    # readAndWrite1(0, os.path.join(root, filePathSrc[0]), os.path.join(root, filePathEnglish[0]), baidu_chinese, english)
    # readAndWrite1(1, os.path.join(root, filePathSrc[1]), os.path.join(root, filePathEnglish[1]), baidu_chinese, english)
    # readAndWrite1(2, os.path.join(root, filePathSrc[2]), os.path.join(root, filePathEnglish[2]), baidu_chinese, english)

    # for i in range(0, fileNum):
    #     readAndWrite1(i, os.path.join(root, filePathSrc[i]),os.path.join(root, filePathEnglish[i]),baidu_chinese,english)

    print("--------------------------end translate english------------------------")
    print("---------------------------start translate chinese----------------------")

    # TODO test chinese to english
    # readAndWrite1(1, os.path.join(root, filePathEnglish[1]), os.path.join(root, filePathChinese[0]), english, baidu_chinese)
    for i in range(0,fileNum):
        readAndWrite1(i, os.path.join(root, filePathEnglish[i]), os.path.join(root, filePathChinese[i]), english, baidu_chinese)
    print("--------------------------end translate chinese------------------------")


if __name__ == '__main__':
    main()
    # print(baidu_translate("中国", 'zh', 'en'))