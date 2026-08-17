from __future__ import annotations

import hashlib
import json
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.hrefs: list[str] = []
        self.title_parts: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(str(values["id"]))
        if tag == "a" and values.get("href"):
            self.hrefs.append(str(values["href"]))
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)


class StaticSiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = (SITE / "index.html").read_text(encoding="utf-8")
        cls.normalized_html = re.sub(r"\s+", " ", cls.html.casefold())
        cls.parser = _PageParser()
        cls.parser.feed(cls.html)

    def test_required_assets_exist(self) -> None:
        for name in (
            "index.html",
            "styles.css",
            "favicon.svg",
            "robots.txt",
            "sitemap.xml",
            "vercel.json",
            "assets/icons/gpu-device.png",
            "assets/icons/ram-device.png",
            "assets/icons/nvme-device.png",
            "assets/icons/arrow-up-bold.svg",
            "assets/icons/LICENSE.phosphor-icons",
            "assets/icons/ASSET_PROVENANCE.md",
            "assets/visuals/moevm-memory-flow-hero.png",
            "assets/visuals/sparse-expert-routing.png",
            "assets/visuals/ASSET_PROVENANCE.md",
        ):
            self.assertTrue((SITE / name).is_file(), name)

    def test_generated_visuals_are_local_and_wired_to_the_page(self) -> None:
        for asset in (
            "assets/visuals/moevm-memory-flow-hero.png",
            "assets/visuals/sparse-expert-routing.png",
        ):
            self.assertEqual(1, self.html.count(f'src="{asset}"'))
            self.assertGreater((SITE / asset).stat().st_size, 100_000)
        self.assertIn('class="hero-showcase"', self.html)
        self.assertIn('class="workflow-visual"', self.html)

    def test_user_supplied_hardware_assets_are_exact(self) -> None:
        expected = {
            "gpu-device.png": "4de39640c607f6608676567f6c697108ccc3568019a5c10db4e2e419b108d71f",
            "ram-device.png": "d94f9de2405d05146715470d1658bf703f522ffa7c5fad82baf6b65f4e0ab8b6",
            "nvme-device.png": "369828eb666f61b2497494815429206b00e750dfb907ea285f1f5f353ecc434b",
        }
        for name, expected_sha256 in expected.items():
            payload = (SITE / "assets" / "icons" / name).read_bytes()
            self.assertEqual(expected_sha256, hashlib.sha256(payload).hexdigest())

    def test_internal_anchors_resolve(self) -> None:
        anchors = [href[1:] for href in self.parser.hrefs if href.startswith("#")]
        self.assertTrue(anchors)
        self.assertEqual(
            [], sorted(anchor for anchor in anchors if anchor not in self.parser.ids)
        )

    def test_public_links_are_https(self) -> None:
        external = [href for href in self.parser.hrefs if not href.startswith("#")]
        self.assertTrue(external)
        self.assertTrue(all(href.startswith("https://") for href in external))

    def test_page_has_no_tracking_or_form_surface(self) -> None:
        lowered = self.html.lower()
        self.assertNotIn("<script", lowered)
        self.assertNotIn("<form", lowered)
        self.assertNotRegex(
            lowered, r"google-analytics|googletagmanager|posthog|segment\.com"
        )

    def test_claim_scope_is_visible_with_measurements(self) -> None:
        for text in (
            "1.338×",
            "1.900×",
            "21.33%",
            "5 prompts",
            "16 teacher-forced tokens each",
            "no concurrency",
            "not a general serving claim",
        ):
            self.assertIn(text.casefold(), self.normalized_html)

    def test_title_and_language_are_set(self) -> None:
        self.assertIn('<html lang="en">', self.html)
        self.assertIn("MoEVM Lab", "".join(self.parser.title_parts))

    def test_vercel_security_headers_are_fail_closed(self) -> None:
        config = json.loads((SITE / "vercel.json").read_text(encoding="utf-8"))
        headers = {
            item["key"]: item["value"] for item in config["headers"][0]["headers"]
        }
        csp = headers["Content-Security-Policy"]
        self.assertIn("script-src 'none'", csp)
        self.assertIn("connect-src 'none'", csp)
        self.assertIn("form-action 'none'", csp)
        self.assertIn("style-src 'self'", csp)
        self.assertNotRegex(self.html, r"<[^>]+\sstyle=")
        self.assertEqual("DENY", headers["X-Frame-Options"])

    def test_demo_requirements_match_guarded_resource_policy(self) -> None:
        for text in (
            "CUDA 13-compatible",
            "8 GiB VRAM total / 4 GiB free",
            "6 GiB recommended",
            "16 GiB RAM / 8 GiB available",
            "up to 35 GiB disk",
        ):
            self.assertIn(text.casefold(), self.normalized_html)

    def test_chart_widths_live_in_external_css(self) -> None:
        css = (SITE / "styles.css").read_text(encoding="utf-8")
        self.assertRegex(css, r"\.bar-baseline\s*\{\s*width:\s*100%;")
        self.assertRegex(css, r"\.bar-empty\s*\{\s*width:\s*74\.73%;")
        self.assertRegex(css, r"\.bar-retained\s*\{\s*width:\s*52\.63%;")

    def test_memory_path_animation_is_css_only_and_motion_safe(self) -> None:
        css = (SITE / "styles.css").read_text(encoding="utf-8")
        for class_name in (
            "expert-path-source",
            "expert-path-staging",
            "expert-path-active",
        ):
            self.assertEqual(1, self.html.count(class_name))
        for keyframes in (
            "@keyframes expert-source-pulse",
            "@keyframes demand-packet",
            "@keyframes lookahead-packet",
            "@keyframes expert-staging-pulse",
            "@keyframes h2d-packet",
            "@keyframes expert-active-pulse",
        ):
            self.assertIn(keyframes, css)
        reduced_motion = css.split("@media (prefers-reduced-motion: reduce)", 1)[1]
        self.assertIn(".memory-system .expert-path-source", reduced_motion)
        self.assertIn(".memory-system .expert-path-staging", reduced_motion)
        self.assertIn(".memory-system .route-packet", reduced_motion)
        self.assertIn("animation: none !important", reduced_motion)
        self.assertIn(".memory-system .motion-control", reduced_motion)
        self.assertGreaterEqual(css.count("4.6s ease-in-out 0.2s infinite both"), 9)
        self.assertIn("animation-play-state: paused", css)
        self.assertIn('id="memory-motion-toggle"', self.html)
        self.assertIn('for="memory-motion-toggle"', self.html)
        self.assertIn("Pause motion", self.html)
        self.assertIn("Play motion", self.html)
        self.assertNotIn("will-change", css)
        self.assertNotIn("<script", self.html.casefold())

    def test_memory_diagram_models_paged_tiers_accurately(self) -> None:
        css = (SITE / "styles.css").read_text(encoding="utf-8")
        self.assertIn(
            'class="memory-system"\n          aria-labelledby="memory-diagram-title"',
            self.html,
        )
        self.assertIn('id="memory-diagram-title"', self.html)
        for asset in (
            "assets/icons/gpu-device.png",
            "assets/icons/ram-device.png",
            "assets/icons/nvme-device.png",
        ):
            self.assertEqual(1, self.html.count(f'src="{asset}"'))
        self.assertEqual(2, self.html.count('class="route route-'))
        self.assertEqual(3, self.html.count('class="route-packet"'))
        self.assertEqual(3, self.html.count('class="legend-line '))
        self.assertNotIn('class="transfer ', self.html)
        self.assertRegex(
            css,
            r"\.route-h2d\s*\{[^}]*--route-x:\s*50%;",
        )
        self.assertRegex(
            css,
            r"\.route-flow-demand\s*\{[^}]*--route-x:\s*34%;",
        )
        self.assertRegex(
            css,
            r"\.route-flow-lookahead\s*\{[^}]*--route-x:\s*68%;",
        )
        self.assertIn(".route-flow-lookahead .route-track", css)
        self.assertIn("repeating-linear-gradient", css)
        self.assertIn('class="route-label">Dedicated H2D stream</span>', self.html)
        self.assertRegex(
            css,
            r"\.route-h2d \.route-label\s*\{[^}]*z-index:\s*3;[^}]*background:\s*var\(--bg\);",
        )
        self.assertIn('class="route-label">Demand</span>', self.html)
        self.assertIn('class="route-label">Lookahead</span>', self.html)
        self.assertIn(
            'class="expert expert-active expert-path-active">E07</span>', self.html
        )
        self.assertNotIn(
            'class="expert expert-active expert-path-active">E01</span>', self.html
        )
        for label in (
            "VRAM expert slots (bounded)",
            "GPU compute",
            "Router · attention · non-expert weights · KV state stay outside paged expert slots.",
            "Bounded pinned RAM staging",
            "not an expert cache",
            "mmap / OS page cache",
            "unobserved by MoEVM",
            "NVMe checkpoint (cold)",
            "Demand (solid)",
            "Lookahead (dotted)",
            "System layer · unobserved",
        ):
            self.assertIn(label, self.html)
        self.assertNotIn("Cache of experts", self.html)
        self.assertNotIn("through the RAM cache", self.html)

    def test_no_private_paths_or_contact_claims(self) -> None:
        self.assertNotRegex(
            self.html, re.compile(r"[A-Za-z]:\\|C:/Users/", re.IGNORECASE)
        )
        self.assertNotIn("hello@moevmlab.com", self.html.casefold())


if __name__ == "__main__":
    unittest.main()
