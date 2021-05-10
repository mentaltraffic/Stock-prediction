# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 19:24:49 2021

@author: HP
"""
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd

import os
from os import path

cwd = os.getcwd()

driver = webdriver.Chrome(path.join(cwd,"chromedriver"))                     #Download chromedriver https://chromedriver.chromium.org/downloads and paste it in this folder
driver.get('https://finance.yahoo.com/most-active')
content = driver.page_source
soup = BeautifulSoup(content,features="html.parser")
# pcount=soup.findAll('a', href=True, attrs={'class':'Fw(600) C($linkColor)})
# print(len(pcount))

for a in soup.findAll('div',attrs={'class':'Ovx(a) Ovx(h)--print Ovy(h) W(100%)'}):
    for link in a.findAll('a', href=True, attrs={'class':'Fw(600) C($linkColor)'}):
        s=link.get_text()
        l='https://finance.yahoo.com/quote/'+s+'/community?p='+s
        print(l)
        driver.get(l)
        # button =driver.find_element_by_css_selector("Show more")
        # button.click()
        button =driver.find_element_by_css_selector("button[class='Fz(16px) Fw(b) Bdw(2px) Ta(c) Cur(p) Va(m) Bdrs(4px) O(n)! Lh(n) Bgc(#fff) C($c-fuji-blue-1-a) Bdc($c-fuji-blue-1-a) Bd C(#fff):h Bgc($c-fuji-blue-1-a):h Mt(20px) Mb(20px) Px(30px) Py(10px) showNext D(b) Mx(a) Pos(r)']")
        button.click()
        button =driver.find_element_by_css_selector("button[class='Fz(16px) Fw(b) Bdw(2px) Ta(c) Cur(p) Va(m) Bdrs(4px) O(n)! Lh(n) Bgc(#fff) C($c-fuji-blue-1-a) Bdc($c-fuji-blue-1-a) Bd C(#fff):h Bgc($c-fuji-blue-1-a):h Mt(20px) Mb(20px) Px(30px) Py(10px) showNext D(b) Mx(a) Pos(r)']")
        button.click()
        con=driver.page_source
        sp=BeautifulSoup(con,features="html.parser")
        f=open(s+".txt","w+",encoding='utf-8')
        for comm in sp.findAll('div',attrs={'class':'C($c-fuji-grey-l) Mb(2px) Fz(14px) Lh(20px) Pend(8px)'}):
            x=comm.get_text()
            f.write(x)
            f.write("\n\n")
        f.close()
            