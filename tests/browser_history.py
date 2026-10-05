"""Explicit-run browser regression using synthetic data; no application server.

Run: uv run --no-sync --with playwright==1.58.0 python tests/browser_history.py
Requires an existing Playwright Chromium installation. Provider calls are mocked.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

# Direct execution puts tests/, not the source-checkout root, on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from fastapi.testclient import TestClient
from playwright.sync_api import expect, sync_playwright

from psxhelper.data import ResearchService, SarmaayaAdapter
from psxhelper.main import create_app


def main():
    adapter = SarmaayaAdapter(client_factory=Mock())
    rows = [
        {"date": "2026-09-01", "price": "2.125", "volume": "0", "marketCap": "0"},
        {"date": "2026-09-02", "price": "2.125", "volume": None, "marketCap": "10"},
        {"date": "2026-09-10", "price": "3", "volume": "5", "marketCap": None},
        {"date": "2026-09-20", "price": "4", "volume": "10", "marketCap": "20"},
        {"date": "2026-10-02", "price": "4.125", "volume": "15", "marketCap": "123456789012345.125"},
    ]
    bundles = {
        symbol: adapter.normalize(symbol, {"symbol": symbol, "name": "Synthetic chart company"}, pd.DataFrame(history))
        for symbol, history in (("DEMO_CHART", rows), ("DEMO_EMPTY", []),
                                ("DEMO_SINGLE", [{**rows[0], "marketCap": None}]))
    }
    provider = Mock()
    provider.fetch.side_effect = lambda symbol: bundles[symbol]
    with TestClient(create_app(ResearchService(provider))) as client, sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            def route_request(route):
                request = route.request
                path = request.url.removeprefix("http://psxhelper.test")
                response = client.request(request.method, path, content=request.post_data)
                route.fulfill(status=response.status_code, body=response.content,
                              content_type=response.headers.get("content-type", "text/plain"))

            context = browser.new_context(viewport={"width": 1180, "height": 900}, has_touch=True)
            context.route("http://psxhelper.test/**", route_request)
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto("http://psxhelper.test/stocks/DEMO_CHART")
            page.wait_for_load_state("networkidle")
            expect(page.locator("#status")).to_contain_text("Loaded DEMO_CHART")
            readout = page.locator("#chart-readout")
            slider = page.locator("#history-position")
            expect(readout).to_contain_text("123456789012345.125")
            expect(slider).to_have_value("4")
            slider.focus()
            page.keyboard.press("Home")
            expect(readout).to_contain_text("2026-09-01")
            expect(readout).to_contain_text("Market cap: 0")
            page.keyboard.press("ArrowRight")
            expect(readout).to_contain_text("Volume: Missing")
            page.keyboard.press("End")
            expect(readout).to_contain_text("2026-10-02")

            def inspect_point(x, touch=False):
                page.locator("#chart").scroll_into_view_if_needed()
                point = page.locator("#chart").evaluate(
                    "(svg,x) => { const p=new DOMPoint(x,100).matrixTransform(svg.getScreenCTM()); return {x:p.x,y:p.y}; }", x)
                if touch:
                    page.touchscreen.tap(point["x"], point["y"])
                else:
                    page.mouse.move(point["x"], point["y"])

            inspect_point(60)
            expect(readout).to_contain_text("2026-09-01")
            page.select_option("#history-view", "market_cap")
            expect(page.locator("#chart polyline")).to_have_count(2)
            slider.focus()
            page.keyboard.press("Home")
            page.keyboard.press("ArrowRight")
            page.keyboard.press("ArrowRight")
            expect(readout).to_contain_text("2026-09-10")
            expect(readout).to_contain_text("Market cap: Missing")
            expect(page.locator("#chart circle").last).to_have_attribute("display", "none")
            inspect_point(60 + 9 / 31 * 800)
            expect(readout).to_contain_text("2026-09-10")
            inspect_point(860)
            expect(readout).to_contain_text("123456789012345.125")
            page.set_viewport_size({"width": 390, "height": 844})
            overflow = page.evaluate("""() => [...document.querySelectorAll('body *')].filter(element => {
                const r=element.getBoundingClientRect(); return !element.closest('.table-wrap') && r.width && r.right > innerWidth;
            }).slice(0,12).map(element => ({tag:element.tagName,id:element.id,width:element.getBoundingClientRect().width}))""")
            if page.evaluate("document.documentElement.scrollWidth > window.innerWidth"):
                print("Mobile diagnostic:", page.evaluate("""() => ({viewport:innerWidth,root:document.documentElement.scrollWidth,
                    body:document.body.scrollWidth,containers:[...document.querySelectorAll('body *')].filter(e => e.scrollWidth > e.clientWidth + 1)
                    .map(e => ({tag:e.tagName,id:e.id,class:e.className,width:e.clientWidth,scroll:e.scrollWidth,overflow:getComputedStyle(e).overflowX})).slice(0,20)})"""))
                page.screenshot(path="/tmp/opencode/psxhelper-history-overflow.png", full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), overflow
            inspect_point(60, touch=True)
            expect(readout).to_contain_text("Market cap: 0")
            inspect_point(860, touch=True)
            expect(readout).to_contain_text("2026-10-02")
            page.screenshot(path="/tmp/opencode/psxhelper-history-mobile.png")

            for symbol in ("DEMO_EMPTY", "DEMO_SINGLE"):
                page.fill("#symbol", symbol)
                page.click("#load")
                expect(page.locator("#status")).to_contain_text("Loaded " + symbol)
                expect(page.locator("#chart polyline")).to_have_count(0)
                if symbol == "DEMO_EMPTY":
                    expect(slider).to_be_disabled()
                    expect(readout).to_have_text("No daily observations available.")
                else:
                    expect(slider).to_be_enabled()
                    expect(readout).to_contain_text("Market cap: Missing")
                    page.select_option("#history-view", "close")
                    expect(page.locator("#chart circle")).to_have_count(2)
            assert not errors, errors
            print("Browser history checks passed: hover, keyboard, touch, gaps, exact readout, empty/single history, ticker reset.")
        finally:
            browser.close()


if __name__ == "__main__":
    main()
