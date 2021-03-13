import facebook
import pandas as pd
import json
import os

dir = os.getcwd()
df = pd.read_excel('Master Modeler Competition 2021.xlsx', usecols='D', names = ['url']) 

token = 'EAADCE73DYfwBAO1uvGQ74ip2DYD97PwJeGyfZBgOWWhaKm15PMNpgcJwTwa1Lk44x3FDzV7wqyzTlnTYzt1WJMRDmByX7sBOrnGdKzFZBDjKxZBtHIHjOjokYXFbXU3ZCtjRPnyp2kAI4TZCq8xjEXeflncMNCqkum9qMgvvwX6ZAOtZA6RkQKJAd3TTYWQ6CjvD2U6uXcuGlQ9cxAbZCq2R4FhVyvXlLkWoLuU2MSZAgQAZDZD'
graph = facebook.GraphAPI(access_token=token)

all_urls = df['url']
unavailable_post = []
finished_urls = []
invalid_fid = []
not_post = []
for i in range(len(all_urls)):
    if i >= 2400:
        furl = all_urls[i]
        print(furl)
        page_id = '1003344626383879'
        if 'photos' in furl:
            post_id = furl[61:77]
        else:
            url = furl.replace('helperase', page_id)
            url = furl.replace('videos', 'posts')
            url = furl.replace('/?substory_index=0', '')
            url = furl.replace(':0', '')
            post_id = url[48:].strip('/')
        if 'https://www.facebook.com/' in furl:
            output_file = open(dir+'/RawData/'+str(i)+'.json', 'w')
        else:
            not_post.append(i)
            continue
        fid = page_id + '_' + post_id
        # meaning of each fileds: https://developers.facebook.com/docs/graph-api/reference/post/
        # for fields not included in the above link, check here: https://developers.facebook.com/docs/graph-api/changelog/version3.3#pages
        try:
            event = graph.get_object(id=fid, fields='created_time,message,message_tags,place,permalink_url,attachments')
            finished_urls.append(i)
            json.dump(event, output_file)
        except facebook.GraphAPIError as e:
            print(str(e))
            if str(e) == '(#100) Pages Public Content Access requires either app secret proof or an app token': 
                unavailable_post.append(i)
            elif str(e) == '(#4) Application request limit reached':
                break
            else:
                invalid_fid.append(i)