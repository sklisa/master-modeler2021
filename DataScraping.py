from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.keys import Keys

url = "https://www.facebook.com/helperase/posts/1628762067175462"

chrome_options = webdriver.ChromeOptions()

#This is how you can make the browser headless
chrome_options.add_argument("--headless")
#The following line controls the notification popping up right after login
prefs = {"profile.default_content_setting_values.notifications" : 2}
chrome_options.add_experimental_option("prefs",prefs)
# driver = webdriver.Chrome(chrome_options=chrome_options)
driver = webdriver.Chrome(executable_path='/Users/lisasun/Desktop/MasterModeler/chromedriver.exe')
driver.get(url)
driver.find_element_by_id("email").send_keys("kelisasun")
driver.find_element_by_id("pass").send_keys("Sk*981004",Keys.RETURN)
driver.get(url)
# soup = BeautifulSoup(driver.page_source, "lxml")
soup = BeautifulSoup(driver.page_source, 'html.parser')
# print(soup)
#
for img in soup.find_all('link',{"as":'image'}):
    print(img.get('href'))
driver.quit()