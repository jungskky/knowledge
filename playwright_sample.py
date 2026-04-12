# pip install playwright
# playwright install

from playwright.sync_api import sync_playwright

def capture_naver_article(url, selector, output_file="naver_article.png"):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            locale="ko-KR",
            timezone_id="Asia/Seoul",
            viewport={"width": 1440, "height": 1600},
        )
        page = context.new_page()

        # networkidle rarely completes on ad-heavy sites (e.g. Naver).
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_selector(selector, state="visible", timeout=30000)

        element = page.locator(selector).first
        element.scroll_into_view_if_needed()
        element.screenshot(path=output_file)

        page.screenshot(path="naver_full.png", full_page=True)

        context.close()
        browser.close()

if __name__ == "__main__":
    capture_naver_article(
        url="https://www.naver.com",
        # 메인 레이아웃에 따라 a.news_tit 대신 뉴스 도메인 링크가 더 안정적일 수 있음
        selector='a[href*="news.naver.com"]',
        output_file="naver_article.png"
    )