import argparse
import os
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-pages", default=100, type=int)
    args = parser.parse_args()
    num_pages = args.num_pages

    dataset_dir = "./space_image_captioning_dataset"
    images_dir = os.path.join(dataset_dir, "./space_images")
    captions_dir = os.path.join(dataset_dir, "./space_captions.txt")
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    caption_file = open(captions_dir, "w", encoding="utf-8")

    WEB_DRIVER_DELAY_TIME = 10
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.headless = True
    driver = webdriver.Chrome(options=chrome_options)
    driver.implicitly_wait(4)
    wait = WebDriverWait(driver, WEB_DRIVER_DELAY_TIME)
    img_idx = 1

    try:
        for page_idx in tqdm(range(1, num_pages + 1)):
            main_url = f"https://kienthuckhoahoc.org/kh-vu-tru/page{page_idx}"
            driver.get(main_url)

            news_lst_xpath = '//a[@class="title"]'
            news_tags = driver.find_elements(By.XPATH, news_lst_xpath)

            news_page_urls = [news_tag.get_attribute("href") for news_tag in news_tags]

            for news_page_url in news_page_urls:
                driver.get(news_page_url)

                img_box_xpath = (
                    '/html/body/div[1]/div[3]/div[1]/div[6]/div[@class="img-box"]'
                )
                img_box_tags = driver.find_elements(By.XPATH, img_box_xpath)

                if img_box_tags:
                    for img_box_tag in img_box_tags:
                        wait = WebDriverWait(img_box_tag, 5)
                        try:
                            img_tag = wait.until(
                                EC.visibility_of_element_located((By.TAG_NAME, "img"))
                            )

                            img_caption = wait.until(
                                EC.visibility_of_element_located((By.TAG_NAME, "em"))
                            )
                        except:
                            continue

                        img_url = img_tag.get_attribute("src")
                        if img_url[-3:] == "gif":
                            continue
                        img_caption = img_caption.text

                        if img_caption == "":
                            continue

                        img_url_resp = requests.get(img_url)
                        try:
                            img = Image.open(BytesIO(img_url_resp.content))
                        except:
                            print(news_page_url)
                            continue

                        if img.mode == "P":
                            img = img.convert("RGB")

                        img_name = f"IMG_{img_idx:06}.jpg"
                        img_save_path = os.path.join(images_dir, img_name)
                        img.save(img_save_path)
                        img_idx += 1

                        caption_file_line_content = (
                            img_save_path + "\t" + img_caption + "\n"
                        )
                        caption_file.write(caption_file_line_content)

                driver.back()
    except Exception as e:
        print(e)
        print(news_page_url)
        pass

    caption_file.close()
