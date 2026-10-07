#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CET 词汇例句批量替换工具（纯本地 / 零联网 / 零外部 API）
=========================================================

功能：
  1. 读取单词列表（或直接从 cet-vocab.html 中提取 3977 个单词）
  2. 加载本地数据源（Tatoeba 离线数据集 或 本地 ECDICT 词典库）
  3. 为每个单词匹配一条真实的中英对照例句
  4. 批量替换 cet-vocab.html 中的占位例句
  5. 带缓存，重复执行不会重复匹配

数据源说明（重要）：
  - Tatoeba  ：提供「英文句子 + 中文译文」对照，是本脚本的例句来源
  - ECDICT   ：本身【不含例句字段】，仅提供音标/释义；若你的 ECDICT 文件
               额外带 example / sentence 之类的列，脚本也会自动识别并使用

支持的数据源文件格式（自动嗅探，无需手动指定）：
  A. Tatoeba 官方导出：sentences.csv (id \t lang \t text) + 同目录的 links.csv
  B. Tatoeba 句对导出：任意 TSV/CSV，两列即「英文 \t 中文」
  C. ECDICT        ：stardict.csv / ecdict.csv，表头含 word / translation
  D. 混合句对      ：四列 id, en_text, id2, zh_text

用法：
  python build_examples.py --dry-run          # 试跑，不写入文件
  python build_examples.py                    # 正式执行
  python build_examples.py --retry-missing    # 只重试上次没匹配到的词
  python build_examples.py --force            # 忽略缓存全部重跑
"""

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time

# ============================ 配置区 ============================
# 一般情况下只需要改这里，或用命令行参数覆盖
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HTML_FILE = os.path.join(BASE_DIR, "cet-vocab.html")           # 待替换的目标文件
WORDS_FILE = os.path.join(BASE_DIR, "words.txt")               # 单词列表（不存在则从 HTML 提取）
SOURCE_FILE = os.path.join(BASE_DIR, "data", "sentences.csv")  # 本地数据源文件
SOURCE_TYPE = "auto"                                           # auto | tatoeba | ecdict | pairs
CACHE_FILE = os.path.join(BASE_DIR, ".example_cache.json")     # 匹配缓存
REPORT_FILE = os.path.join(BASE_DIR, "match_report.txt")       # 匹配报告
UNMATCHED_FILE = os.path.join(BASE_DIR, "unmatched_words.txt") # 未匹配单词清单

MAKE_BACKUP = True            # 覆盖前是否备份原 HTML（.bak）
FALLBACK = "keep"             # 匹配失败时的降级方案：keep=保留原占位符 | template=生成兜底句
SIMPLIFY_CN = True            # 是否把繁体译文转换为简体（依赖 zhconv，缺失时自动跳过）

# 例句质量过滤参数
MIN_WORDS = 4                 # 英文例句最少词数
MAX_WORDS = 20                # 英文例句最多词数
MAX_CANDIDATES = 600          # 每个单词最多考察的候选句子数（控制耗时）
# ===============================================================


# ------------------------------------------------------------------
# 基础工具
# ------------------------------------------------------------------
def log(msg):
    print(msg, flush=True)


def read_lines(path):
    """按多种编码尝试读取文本行，兼容 UTF-8 / UTF-8-BOM / GBK"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                return f.read().splitlines(), enc
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"无法解码文件（尝试过 utf-8/utf-8-sig/gbk）: {path}")


def sniff_delimiter(first_line):
    """嗅探分隔符：制表符优先，其次逗号"""
    if "\t" in first_line:
        return "\t"
    if "," in first_line:
        return ","
    return "\t"


def open_csv(path, delimiter):
    """打开 CSV/TSV，正确处理引号"""
    return open(path, "r", encoding="utf-8", errors="replace", newline="")


def js_escape(s):
    """转义为可安全嵌入 JS 双引号字符串的内容"""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\r", "")
    s = s.replace("\n", "\\n")
    s = s.replace("\t", "\\t")
    return s


def js_unescape(s):
    """把 HTML 中读到的 JS 字符串还原为原始文本"""
    if s is None:
        return ""
    return s.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")


def has_cjk(s):
    """是否包含中日韩文字，用于校验中文译文非空且有效"""
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)


# 繁 -> 简 转换（zhconv 为纯 Python 本地库，不联网；未安装时自动降级为不转换）
try:
    from zhconv import convert as _zh_convert

    def to_simplified(s):
        return _zh_convert(s, "zh-cn")
except ImportError:  # pragma: no cover
    def to_simplified(s):
        return s


def word_count_en(s):
    return len(re.findall(r"[A-Za-z][A-Za-z'\-]*", s))


def stem(tok):
    """
    极简英文词干化，用于倒排索引，保证「原形 ↔ 屈折形」能互相命中。
    例：eradicate / eradicated / eradicating  ->  eradicat
        run / running                         ->  run
        study / studies                       ->  study
    只做工程上够用的规则，不追求语言学精确。
    """
    t = tok.lower().strip("'-")
    if len(t) <= 3:
        return t
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("ied") and len(t) > 4:
        return t[:-3] + "y"
    stripped = False
    for suf in ("ingly", "edly", "ing", "ed"):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            t = t[:-len(suf)]
            stripped = True
            break
    if not stripped:
        for suf in ("es", "s"):
            if t.endswith(suf) and len(t) - len(suf) >= 3:
                t = t[:-len(suf)]
                break
    # 去重复辅音：runn -> run
    if len(t) >= 4 and t[-1] == t[-2] and t[-1] not in "aeiou":
        t = t[:-1]
    # 去尾部 e：eradicat(e) -> eradicat
    if len(t) >= 4 and t.endswith("e"):
        t = t[:-1]
    return t


# ------------------------------------------------------------------
# 句式匹配：为单词/词组构建可容忍词形变化的正则
# ------------------------------------------------------------------
_SUFFIX = r"(?:s|es|ed|d|ing|ly)?"


def build_pattern(term):
    """
    把单词或词组编译成正则。
    规则：
      - 单词：允许常见屈折后缀（-s/-es/-ed/-ing/-ly）
      - 词组：各成分允许屈折，成分间允许任意空白
      - 首字母大小写不敏感，但不允许嵌在其他字母中间
    返回 (regex, [索引键, ...])，索引键为各成分词干化后的结果
    """
    tokens = re.findall(r"[A-Za-z][A-Za-z'\-]*", term)
    if not tokens:
        return None, None

    parts = []
    for t in tokens:
        # 处理辅音重复：run -> running / plan -> planning
        body_t = re.escape(t)
        last = t[-1].lower()
        if len(t) >= 3 and last not in "aeiou" and last.isalpha():
            body_t += r"(?:" + re.escape(t[-1]) + r")?"
        parts.append(body_t + _SUFFIX)
    body = r"\s+".join(parts)

    # 左右都要是非字母边界，避免 "art" 命中 "start"
    pattern = re.compile(r"(?<![A-Za-z])" + body + r"(?![A-Za-z])", re.IGNORECASE)

    # 索引键：各 token 的词干（去重、保序）
    keys = []
    for t in tokens:
        k = stem(re.sub(r"[^A-Za-z]", "", t))
        if len(k) >= 2 and k not in keys:
            keys.append(k)
    return pattern, keys


# ------------------------------------------------------------------
# 数据源加载
# ------------------------------------------------------------------
class ExampleSource:
    """持有全部候选句对，并用倒排索引加速检索"""

    def __init__(self):
        self.pairs = []                 # [(en, zh), ...]
        self.index = {}                 # token -> [pair_index, ...]
        self.ecdict = {}                # word -> {"phonetic":..., "translation":...}

    def add_pair(self, en, zh):
        en = (en or "").strip()
        zh = (zh or "").strip()
        if not en or not zh or not has_cjk(zh):
            return
        if not (MIN_WORDS <= word_count_en(en) <= MAX_WORDS):
            return
        idx = len(self.pairs)
        self.pairs.append((en, zh))
        # 倒排索引建立在词干上，保证原形/屈折形能互相命中
        for tok in set(re.findall(r"[A-Za-z][A-Za-z'\-]*", en)):
            k = stem(re.sub(r"[^A-Za-z]", "", tok))
            if len(k) >= 2:
                self.index.setdefault(k, []).append(idx)

    def stats(self):
        return len(self.pairs)


def detect_source_type(path, first_line):
    """根据首行内容判断数据源类型"""
    low = first_line.lower()
    if "word" in low and ("translation" in low or "definition" in low or "phonetic" in low):
        return "ecdict"
    cols = first_line.split(",") if "," in first_line and "\t" not in first_line else first_line.split("\t")
    # sentences.csv: id, lang, text  → 首列数字、次列三字母语言码
    if len(cols) >= 3 and cols[0].strip().strip('"').isdigit() and re.fullmatch(r"[a-z]{3}", cols[1].strip().strip('"')):
        return "tatoeba"
    return "pairs"


def load_source(path, forced_type="auto"):
    """加载数据源，返回 ExampleSource"""
    src = ExampleSource()

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"数据源文件不存在: {path}\n"
            f"请把 Tatoeba / ECDICT 数据文件放到该路径，或用 --source 指定。"
        )

    lines, enc = read_lines(path)
    if not lines:
        raise RuntimeError(f"数据源文件为空: {path}")

    first = lines[0]
    stype = forced_type if forced_type != "auto" else detect_source_type(path, first)
    log(f"  数据源: {path}")
    log(f"  编码: {enc}   类型: {stype}   首行: {first[:80]!r}")

    if stype == "ecdict":
        load_ecdict(src, path)
    elif stype == "tatoeba":
        load_tatoeba_sentences(src, path, lines)
    else:
        load_pairs(src, path, lines)

    return src


def load_ecdict(src, path):
    """加载 ECDICT。注意：ECDICT 无例句字段，这里用于补充音标/释义；
    若文件中存在 example/sentence 列，则同时收录为句对。"""
    with open_csv(path, ",") as f:
        reader = csv.DictReader(f)
        fields = [(fld or "").strip().lower() for fld in (reader.fieldnames or [])]
        for i, row in enumerate(reader):
            # 兼容 BOM
            norm = {(k or "").strip().lower(): (v or "") for k, v in row.items()}
            word = norm.get("word", "").strip()
            if not word:
                continue
            src.ecdict[word.lower()] = {
                "phonetic": norm.get("phonetic", "").strip(),
                "translation": norm.get("translation", "").strip(),
                "definition": norm.get("definition", "").strip(),
            }
            # 若该 ECDICT 扩展版带例句列，则尝试收录
            ex_en = norm.get("example", "") or norm.get("sentence", "") or norm.get("example_en", "")
            ex_zh = norm.get("example_cn", "") or norm.get("sentence_cn", "") or norm.get("example_translation", "")
            if ex_en and ex_zh:
                src.add_pair(ex_en, ex_zh)
    log(f"  ECDICT 词条: {len(src.ecdict)}，例句句对: {src.stats()}")


def load_tatoeba_sentences(src, path, lines):
    """加载 Tatoeba sentences.csv，并自动寻找同目录 links.csv 建立中英对照"""
    delim = sniff_delimiter(lines[0])
    en_text = {}
    zh_text = {}
    mapping = {}

    for line in lines:
        cols = line.split(delim)
        if len(cols) < 3:
            continue
        sid = cols[0].strip().strip('"')
        lang = cols[1].strip().strip('"').lower()
        text = delim.join(cols[2:]).strip().strip('"')
        if not sid.isdigit() or not text:
            continue
        if lang in ("eng", "en"):
            en_text[sid] = text
        elif lang in ("cmn", "zh", "zho", "zh-cn", "zh-tw"):
            zh_text[sid] = text

    log(f"  英文句: {len(en_text)}   中文句: {len(zh_text)}")

    # 寻找 links.csv（同目录）
    dirname = os.path.dirname(os.path.abspath(path))
    links_path = os.path.join(dirname, "links.csv")
    if not os.path.exists(links_path):
        for name in os.listdir(dirname):
            if name.lower().startswith("link") and name.lower().endswith((".csv", ".tsv")):
                links_path = os.path.join(dirname, name)
                break

    if os.path.exists(links_path):
        log(f"  关联文件: {links_path}")
        with open(links_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                cols = line.rstrip("\n").split("\t" if "\t" in line else ",")
                if len(cols) < 2:
                    continue
                a, b = cols[0].strip().strip('"'), cols[1].strip().strip('"')
                if a in en_text and b in zh_text:
                    mapping.setdefault(a, b)
                elif b in en_text and a in zh_text:
                    mapping.setdefault(b, a)
        log(f"  中英句对: {len(mapping)}")
    else:
        log("  ⚠ 未找到 links.csv，无法建立中英对照。")
        log("    请一并下载 Tatoeba 的 links.csv 放到同一目录，")
        log("    或改用「句对」格式的数据文件（两列：英文<TAB>中文）。")
        return

    for en_id, zh_id in mapping.items():
        src.add_pair(en_text[en_id], zh_text[zh_id])
    log(f"  可用句对: {src.stats()}")


def load_pairs(src, path, lines):
    """加载通用句对文件：
       - 两列：(英文, 中文) 或 (中文, 英文)
       - 四列：(id, 文本, id, 文本)，中英顺序不限
    通过「是否含中文字符」判定语向，不依赖列位置。
    """
    delim = sniff_delimiter(lines[0])
    skipped = 0
    for i, line in enumerate(lines):
        cols = [c.strip().strip('"').lstrip("\ufeff") for c in line.split(delim)]
        if len(cols) < 2:
            skipped += 1
            continue
        if i == 0 and re.search(r"english|chinese|sentence|英文|中文",
                                " ".join(cols), re.IGNORECASE) and not has_cjk(cols[-1]):
            continue

        if len(cols) >= 4 and cols[0].isdigit() and cols[2].isdigit():
            a, b = cols[1], cols[3]
        else:
            a, b = cols[0], cols[1]

        # 按内容判定语向
        a_cjk, b_cjk = has_cjk(a), has_cjk(b)
        a_en = bool(re.search(r"[A-Za-z]{2}", a))
        b_en = bool(re.search(r"[A-Za-z]{2}", b))
        if a_cjk and b_en and not b_cjk:
            en, zh = b, a
        elif b_cjk and a_en and not a_cjk:
            en, zh = a, b
        else:
            skipped += 1
            continue
        src.add_pair(en, zh)
    if skipped:
        log(f"  跳过无效行: {skipped}")
    log(f"  可用句对: {src.stats()}")


# ------------------------------------------------------------------
# 匹配核心
# ------------------------------------------------------------------
def score_sentence(en, pattern, term):
    """给候选例句打分，分数越高越优先"""
    score = 0.0
    n = word_count_en(en)
    # 长度偏好：8~14 词最佳
    if 8 <= n <= 14:
        score += 3.0
    elif 5 <= n <= 17:
        score += 1.5
    else:
        score -= 0.5
    score -= abs(n - 11) * 0.08                # 越接近 11 词越好
    # 完整句（首字母大写 + 句末标点）加分
    if en[:1].isupper():
        score += 0.5
    if en.rstrip()[-1:] in ".!?":
        score += 0.5
    # 精确词形（未屈折）加分
    if re.search(r"(?<![A-Za-z])" + re.escape(term) + r"(?![A-Za-z])", en, re.IGNORECASE):
        score += 2.0
    # 含引号/特殊符号轻微减分
    if any(c in en for c in ['"', "…", "—", ";"]):
        score -= 0.3
    # 词组：完整出现加分
    if " " in term and pattern.search(en):
        score += 1.0
    return score


def find_example(src, term, used_sentences):
    """为一个单词/词组寻找最佳例句，返回 (en, zh, pair_index) 或 None"""
    pattern, keys = build_pattern(term)
    if pattern is None or not keys:
        return None

    # 用倒排索引取候选：优先选倒排链最短的键（检索集最小、最快）
    posting_lists = [src.index[k] for k in keys if k in src.index]
    if not posting_lists:
        return None
    posting_lists.sort(key=len)
    candidates = posting_lists[0][:MAX_CANDIDATES]

    best = None
    best_score = float("-inf")
    for idx in candidates:
        if idx in used_sentences:
            continue
        en, zh = src.pairs[idx]
        if not pattern.search(en):
            continue
        s = score_sentence(en, pattern, term)
        if s > best_score:
            best_score = s
            best = (en, zh, idx)
    if best is None:
        return None
    return best[0], best[1], best[2]


# ------------------------------------------------------------------
# 缓存
# ------------------------------------------------------------------
def load_cache(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("words", {}) if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as e:
            log(f"  ⚠ 缓存文件损坏，将重建: {e}")
    return {}


def save_cache(path, words):
    payload = {"version": 1, "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "words": words}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# ------------------------------------------------------------------
# HTML 读写
# ------------------------------------------------------------------
PLACEHOLDER_MARK = "This is a sentence with the word"

ENTRY_RE = re.compile(
    r'\{word:"((?:[^"\\]|\\.)*)",'
    r'phonetic:"((?:[^"\\]|\\.)*)",'
    r'meaning:"((?:[^"\\]|\\.)*)",'
    r'example:"((?:[^"\\]|\\.)*)",'
    r'exampleCn:"((?:[^"\\]|\\.)*)",'
    r'emoji:"((?:[^"\\]|\\.)*)"\}'
)


def extract_entries(html):
    """提取 WORD_LIST 中所有词条及其位置"""
    return [(m.group(1), m.start(), m.end(), m.groups()) for m in ENTRY_RE.finditer(html)]


def extract_words_from_html(html):
    return [js_unescape(m.group(1)) for m in ENTRY_RE.finditer(html)]


def read_words(path, html_path):
    """优先读单词列表文件；不存在则从 HTML 提取"""
    if path and os.path.exists(path):
        lines, enc = read_lines(path)
        words = []
        for line in lines:
            w = line.strip()
            if not w or w.startswith("#"):
                continue
            words.append(w)
        log(f"  单词列表文件: {path}（编码 {enc}）")
        return words
    log(f"  单词列表文件不存在，改为从 HTML 提取: {html_path}")
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    return extract_words_from_html(html)


def replace_in_html(html, cache):
    """把匹配结果写回 WORD_LIST，返回 (新HTML, 命中数, 占位符残留数)"""
    replaced = 0
    remaining_placeholder = 0

    def _sub(m):
        nonlocal replaced, remaining_placeholder
        word = js_unescape(m.group(1))
        item = cache.get(word) or cache.get(word.lower())

        if item and item.get("status") == "ok" and item.get("example"):
            en = js_escape(item["example"])
            zh = js_escape(item.get("exampleCn", ""))
            replaced += 1
        else:
            en = m.group(4)
            zh = m.group(5)
            if PLACEHOLDER_MARK in js_unescape(en):
                remaining_placeholder += 1

        return (
            '{word:"' + m.group(1) + '",'
            'phonetic:"' + m.group(2) + '",'
            'meaning:"' + m.group(3) + '",'
            'example:"' + en + '",'
            'exampleCn:"' + zh + '",'
            'emoji:"' + m.group(6) + '"}'
        )

    new_html = ENTRY_RE.sub(_sub, html)
    return new_html, replaced, remaining_placeholder


def guess_fallback(term):
    """降级兜底：生成一条简单可用的例句（仅在 FALLBACK='template' 时使用）"""
    safe = term.replace('"', "'")
    return f"I learned the word '{safe}' today.", f"我今天学了“{safe}”这个词。"


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="CET 词汇例句批量替换工具（纯本地，不联网）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--html", default=HTML_FILE, help="目标 HTML 文件路径")
    parser.add_argument("--words", default=WORDS_FILE, help="单词列表文件（每行一个），缺省则从 HTML 提取")
    parser.add_argument("--source", default=SOURCE_FILE, help="本地数据源文件路径")
    parser.add_argument("--type", default=SOURCE_TYPE,
                        choices=["auto", "tatoeba", "ecdict", "pairs"], help="数据源类型")
    parser.add_argument("--cache", default=CACHE_FILE, help="缓存文件路径")
    parser.add_argument("--report", default=REPORT_FILE, help="报告输出路径")
    parser.add_argument("--unmatched", default=UNMATCHED_FILE, help="未匹配单词清单路径")
    parser.add_argument("--dry-run", action="store_true", help="只试跑，不写入任何文件")
    parser.add_argument("--force", action="store_true", help="忽略缓存，全部重新匹配")
    parser.add_argument("--retry-missing", action="store_true", help="只重新匹配上次失败的单词")
    args = parser.parse_args()

    t0 = time.time()
    log("=" * 62)
    log("CET 词汇例句批量替换工具")
    log("=" * 62)

    # ---- 前置校验 ----
    if not os.path.exists(args.html):
        log(f"✗ 目标 HTML 不存在: {args.html}")
        return 1
    if not os.path.exists(args.source):
        log(f"✗ 数据源文件不存在: {args.source}")
        log("")
        log("请按以下步骤准备数据文件：")
        log(f"  1) 新建目录: {os.path.dirname(args.source)}")
        log(f"  2) 放入 Tatoeba 的 sentences.csv 与 links.csv")
        log(f"     （或放入任意「英文<TAB>中文」两列句对文件，改 --type pairs）")
        log(f"  3) 重新运行本脚本")
        return 1

    # ---- 1. 读取单词 ----
    log("\n[1/5] 读取单词列表")
    words = read_words(args.words, args.html)
    words = list(dict.fromkeys(words))          # 去重保序
    log(f"  待处理单词: {len(words)} 个")

    # ---- 2. 加载数据源 ----
    log("\n[2/5] 加载本地数据源")
    try:
        src = load_source(args.source, args.type)
    except Exception as e:
        log(f"✗ 数据源加载失败: {e}")
        return 1
    if src.stats() == 0 and not src.ecdict:
        log("✗ 数据源中没有可用的中英句对，请检查文件是否正确。")
        return 1

    # ---- 3. 逐词匹配（带缓存）----
    log("\n[3/5] 匹配例句")
    cache = {} if args.force else load_cache(args.cache)
    cache_hit, matched, missing, reused = 0, 0, 0, 0
    used_sentences = set()

    # 先占位已缓存句子的索引，尽量避免重复使用同一句
    for w, item in cache.items():
        if item.get("status") == "ok":
            cache_hit += 1

    todo = []
    for w in words:
        item = cache.get(w)
        if item and item.get("status") == "ok" and not args.force:
            continue
        if item and item.get("status") == "missing" and not (args.force or args.retry_missing):
            missing += 1
            continue
        todo.append(w)

    log(f"  缓存命中跳过: {cache_hit}   本次待匹配: {len(todo)}")

    def match_pass(word_list, used, label):
        """对一批单词做匹配；used 为句子互斥集合，命中后写入缓存并返回命中数"""
        got = 0
        for i, w in enumerate(word_list, 1):
            try:
                hit = find_example(src, w, used)
            except re.error as e:
                log(f"  ⚠ 正则构建失败 [{w}]: {e}")
                hit = None
            except Exception as e:
                log(f"  ⚠ 匹配异常 [{w}]: {type(e).__name__} {e}")
                hit = None
            if hit:
                en, zh, idx = hit
                used.add(idx)
                if SIMPLIFY_CN:
                    zh = to_simplified(zh)
                cache[w] = {"status": "ok", "example": en, "exampleCn": zh, "source": args.type}
                got += 1
            if i % 300 == 0 and i < len(word_list):
                log(f"    {label} {i}/{len(word_list)}   已命中 {got}")
        return got

    # 第一轮：句子互斥，保证每个词拿到不同例句
    first = match_pass(todo, used_sentences, "第一轮")

    # 第二轮：仅针对第一轮失败的词，放宽句子互斥限制再试一次（仍保证批内互斥）
    missed_now = [w for w in todo if cache.get(w, {}).get("status") != "ok"]
    rescued = 0
    if missed_now:
        log(f"  第二轮：对 {len(missed_now)} 个未命中词放宽限制重试")
        rescued = match_pass(missed_now, set(), "第二轮")

    matched += first + rescued

    # 仍未命中的，按降级方案处理
    for w in missed_now:
        if cache.get(w, {}).get("status") == "ok":
            continue
        if FALLBACK == "template":
            en, zh = guess_fallback(w)
            cache[w] = {"status": "fallback", "example": en, "exampleCn": zh, "source": "template"}
            reused += 1
        else:
            cache[w] = {"status": "missing", "example": "", "exampleCn": "", "source": ""}
            missing += 1

    log(f"  第一轮命中 {first}   第二轮救回 {rescued}   合计命中 {matched}")

    # ---- 4. 写回 HTML ----
    log("\n[4/5] 写入 HTML")
    with open(args.html, "r", encoding="utf-8") as f:
        html = f.read()
    new_html, replaced, leftover = replace_in_html(html, cache)

    log(f"  实际替换词条: {replaced}")
    log(f"  仍为占位例句: {leftover}")

    if args.dry_run:
        log("  （--dry-run 模式，未写入任何文件）")
    else:
        if MAKE_BACKUP:
            bak = args.html + ".bak"
            shutil.copy2(args.html, bak)
            log(f"  已备份原文件: {bak}")
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(new_html)
        save_cache(args.cache, cache)
        log(f"  已写入: {args.html}")
        log(f"  已保存缓存: {args.cache}")

        # 未匹配清单
        unmatched = sorted([w for w, v in cache.items() if v.get("status") == "missing"])
        if unmatched:
            with open(args.unmatched, "w", encoding="utf-8") as f:
                f.write("\n".join(unmatched))
            log(f"  未匹配单词已导出: {args.unmatched}（{len(unmatched)} 个）")

    # ---- 5. 报告 ----
    elapsed = time.time() - t0
    size_mb = os.path.getsize(args.html) / 1024 / 1024
    lines = [
        "=" * 62,
        "CET 例句替换报告",
        "=" * 62,
        f"执行时间      : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"耗时          : {elapsed:.1f} 秒",
        f"数据源        : {args.source}",
        f"数据源句对    : {src.stats()}",
        f"单词总数      : {len(words)}",
        f"缓存命中      : {cache_hit}",
        f"新匹配成功    : {matched}",
        f"降级兜底      : {reused}",
        f"未匹配        : {missing}",
        f"HTML 替换条数 : {replaced}",
        f"占位残留      : {leftover}",
        f"HTML 文件大小 : {size_mb:.2f} MB",
        "=" * 62,
    ]
    report = "\n".join(lines)
    log("\n[5/5] 完成")
    log(report)

    if not args.dry_run:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        log(f"报告已保存: {args.report}")

    if size_mb > 3:
        log(f"\n⚠ 提示：HTML 已达 {size_mb:.2f} MB，部分浏览器打开可能变慢。")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("\n已中断。缓存已保留，可用 --retry-missing 继续。")
        sys.exit(130)
    except Exception as exc:
        log(f"\n✗ 未预期的错误: {type(exc).__name__}: {exc}")
        sys.exit(1)
