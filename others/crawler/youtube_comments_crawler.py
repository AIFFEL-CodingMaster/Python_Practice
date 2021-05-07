import os
import re
import time
import argparse
import pandas as pd
from bs4 import BeautifulSoup
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys


def pagedown_to_bottom(time_setting=3):
    '''
    time_setting: 빨리 눌러서 씹힘 방지 웨이팅 시간(초)
    '''
    scroll_pause_time = time_setting
    last_height = browser.execute_script(
        "return document.documentElement.scrollHeight")
    while True:
        # 스크롤 끝까지 내리기
        browser.execute_script(
            "window.scrollTo(0,document.documentElement.scrollHeight);")
        # 로딩 기다리기 // 너무 빨리 돌리면 씹힘
        time.sleep(scroll_pause_time)
        # 새로운 높이와 갱신된 높이를 비교해서 같으면 종료 => 같다는 것은 더 이상 갱신이 안된다는 뜻
        new_height = browser.execute_script(
            "return document.documentElement.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


def pagedown(page_down_num=1, time_setting=2):
    '''
    page_down_num: 페이지 다운 클릭 횟수
    time_setting: 빨리 눌러서 씹힘 방지 웨이팅 시간(초)
    '''
    body = browser.find_element_by_tag_name('body')
    while page_down_num:
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(time_setting)
        page_down_num -= 1


def youtube_video_url_crawler(limit_num=10):
    '''특정 유튜버의 동영상 카테고리의 url이어야함'''
    html = browser.page_source
    src = BeautifulSoup(html, 'html.parser')

    video_url = src.find_all('a', attrs={
                             'id': 'thumbnail', 'class': 'yt-simple-endpoint inline-block style-scope ytd-thumbnail'})
    urls = []
    length = 0
    for row in video_url:
        if row.get('href') != None:
            address = row.get('href')
            true_address = 'https://www.youtube.com' + address
            urls.append(true_address)
            length += 1
        if length >= limit_num:
            break
    return urls


def youtube_comment_crawler():
    ''' 
    댓글 크롤링 시작
    ##댓글의 댓글은 크롤링 안됨##
    '''
    # 데이터 프레임 안에 저장
    comment_data = pd.DataFrame({'youtube_ID': [],
                                 'cmt': [],
                                 'like_num': []})
    # html source
    html = browser.page_source
    src = BeautifulSoup(html, 'html.parser')

    # 댓글 찾기 -> html tag와 class로 찾음
    comment = src.find_all(
        'ytd-comment-renderer', attrs={'class': 'style-scope ytd-comment-thread-renderer'})

    # 댓글의 전체 길이만큼 반복하여 댓글 추출
    for i in range(len(comment)):
        # 댓글
        comment0 = comment[i].find(
            'yt-formatted-string', {'id': 'content-text', 'class': 'style-scope ytd-comment-renderer'}).text
        # 좋아요
        try:
            aa = comment[i].find('span', {'id': 'vote-count-left'}).text

            like_num = "".join(re.findall('[0-9]', aa))+"개"
        except:
            like_num = 0
        # 아이디
        bb = comment[i].find('a', {'id': 'author-text'}).find('span').text
        youtube_id = ''.join(re.findall('[가-힣0-9a-zA-Z]', bb))
        # 아이디, 댓글, 좋아요 column별로 DataFrame만들기
        insert_data = pd.DataFrame({'youtube_ID': youtube_id,
                                    'cmt': [comment0],
                                    'like_num': [like_num]})
        comment_data = comment_data.append(insert_data)
    comment_data = comment_data.reset_index(drop=True)
    return comment_data


def to_bool(x):
    if x == 'true':
        return True
    elif x == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError("Please input true or false")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chromedriver", type=str,
                        default="", help="chromedriver 저장 위치")
    parser.add_argument("--many_video", type=to_bool,
                        default="false", help="여러개의 영상에서 댓글을 가져올 것인지의 여부")
    parser.add_argument("--save_name", type=str,
                        default="coments.csv", help="저장될 파일명")
    parser.add_argument("--url", type=str, default="", help="url 입력")
    parser.add_argument("--time_setting", type=int,
                        default=2, help="씹힘 방지 기다림 시간")
    parser.add_argument("--video_nums", type=int,
                        default=10, help="가져올 video의 수")
    parser.add_argument("--scrolldown_nums", type=int,
                        default=15, help="스크롤바를 몇 번 내릴 것인지?")
    parser.add_argument("--down_to_bottom", type=to_bool,
                        default="false", help="페이지를 끝까지 내릴 것인가?")
    args = parser.parse_args()

save_dir = "./comments/"
# comments 디렉토리가 없으면 생성
if os.path.exists(save_dir) is False:
    os.mkdir(save_dir)

# 맨 아래까지 페이지를 내릴 것인지 여부


def is_bottom(bottom):
    if bottom is True:
        pagedown_to_bottom(args.time_setting)
    else:
        pagedown(args.scrolldown_nums, args.time_setting)


# Option - window size를 줄여서 bottom으로 내려도 댓글이 잘 나오게 만듬
option = Options()
# 500 x 800 size
option.add_argument("--window-size=500,800")

browser = Chrome(args.chromedriver, options=option)
browser.implicitly_wait(3)
# url open
browser.get(args.url)
browser.implicitly_wait(3)

if args.many_video is True:
    urls = youtube_video_url_crawler(args.video_nums)
    for i, u in enumerate(urls):
        browser.get(u)
        is_bottom(args.down_to_bottom)
        commentary = youtube_comment_crawler()
        commentary.to_csv(save_dir+str(i+1)+args.save_name, index=False)
        print(f"{i+1}번째 영상에서 댓글을 추출 했습니다.")
        print("5초간 휴식")
        time.sleep(5)
else:
    is_bottom(args.down_to_bottom)
    commentary = youtube_comment_crawler()
    commentary.to_csv(save_dir+args.save_name, index=False)
    print("영상에서 댓글을 추출 했습니다.")
