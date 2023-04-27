import requests#导入请求库
import time
import re
url='https://image.baidu.com/search/acjson?tn=resultjson_com&logid=12009735572442623815&ipn=rj&ct=201326592&is=&fp=result&fr=&word=%E5%94%90%E5%AB%A3&cg=star&queryWord=%E5%94%90%E5%AB%A3&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=0&hd=&latest=&copyright=&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&expermode=&nojc=&isAsync=&pn=30&rn=30&gsm=1e&1678976552414='
#添加请求头，模拟浏览器，有些网站可以不加这个，不过最好是加上，油多不坏菜这个道理
headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36'}
res=requests.get(url,headers=headers)#发送请求，返回数据
html=res.text#把返回的内容解析
# 使用正则表达式匹配图片url
img_url_list=re.findall('"thumbURL":"(.*?)"',html)
#print(img_url_list)
for i in range(len(img_url_list)):
    res_img=requests.get(img_url_list[i],headers=headers)
    img=res_img.content#这个里是图片，我们需要返回二进制数据
    # 把图片保存起来
    with open(str(i)+'tangyan_img.jpg','wb')as f:
        f.write(img)
    time.sleep(3)#每当保存一张图片，先暂停一下，不然太频繁容易发现是机器爬虫，导致无法获取

print("爬取{}张图片成功".format(i))

