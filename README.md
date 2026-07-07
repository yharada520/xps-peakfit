# xps-peakfit

**BICによるピーク本数の自動推定 ＋ MAP推定 ＋ 物理拘束スピン軌道ダブレット** を組み合わせた、XPS自動ピーク分離パッケージ。

*[English README is available here / 英語版READMEはこちら → README.en.md](README.en.md)*

金属基板上の急峻な背景に弱い化学状態ショルダーが乗るような難スペクトル（例: Au電極上シロキサンのSi 2p、Auゴースト線との重畳）でも、範囲と線種を指定するだけで物理的に妥当な分解を自動選択することを目標に設計しています。

## 特徴

- **擬Voigtピーク**（面積式は数値積分照合テストで検証済み）と **Doniach-Šunjić形状**（金属の非対称ピーク向け）
- **Active background**: Shirley / Tougaard背景を「現在のピーク成分和」からフィット内部で計算。背景前引き方式で起こる背景誤差→面積誤差の伝播を排除
- **背景モデル自体もBICで自動選択**（`background="auto"`）: Shirley か Tougaard かをデータに決めさせる
- **ラインデータベースによる物理拘束**: スピン軌道分裂・面積比（Si2p: Δ0.61 eV 2:1、Au4f: Δ3.67 eV 4:3 等）を厳密に適用。同一ラインの化学状態間ではFWHM・ηを共有
- **スピン軌道定数の微変動**（`vary_so=True`）: 分裂幅±0.02 eV・分岐比±0.02のタイトなガウス事前分布下で微調整可能（装置差の吸収）
- **MAP推定**: 文献の化学シフト位置を中心のガウス事前分布として使用。事前分布ペナルティはモデル比較にも算入されるため、物理的に不自然な解は順位が下がる
- **BICモデル選択と事後確率表示**: 候補構成（物理プールの部分集合、または無拘束1..N本）を総当たりし、近似事後確率で報告（例: 「n=3: 60%、n=2: 40%」）。ΔBIC僅差の候補は隠さず併記
- **ノイズスケール校正**: 2階差分MADによる頑健推定で、平均化・スケール済みデータでもBICが機能
- **適応マルチスタート**（解一致で早期終了、約5倍高速化）+ **スペクトル適応EM初期値生成**（無拘束モード）
- **ベイズ不確かさ定量化**（opt-in, `emcee`）: 中心・幅・面積の68%信用区間をMAP事後分布そのもののMCMCで取得。数十秒オーダー
- **フィット範囲の自動検出**（ユーザー調整可能）
- Streamlit GUI ＋ バッチCLI

## インストール

```bash
pip install -e .[gui,bayes,dev]
```

## クイックスタート

### CLI

```bash
# 物理拘束モード: Si2pの全化学状態 + 99.8 eV付近のAu4f形状ゴースト
xps-peakfit data/XPS_Si2p_siloxane_synthetic.csv \
    --window 98 104.5 --background auto \
    --line Si2p --ghost Au4f@99.8:0.5 --min-components 2

# 無拘束モード: 最大5本の独立擬Voigt、範囲自動検出、MCMC信頼区間付き
xps-peakfit data/XPS_Si2p.csv --auto-range --generic 5 --bayes
```

### GUI

```bash
streamlit run app_streamlit.py
```

CSV（`energy`,`int`列。別名も柔軟対応）をアップロードし、自動検出された範囲を調整、線種・化学状態を選んで実行。ベストフィット分解・BIC比較表・成分数事後確率・CSVダウンロード・emcee信頼区間が得られます。

### Python API

```python
from xps_peakfit import load_spectrum, select_model
from xps_peakfit.fitting import Component
from xps_peakfit.model_select import subset_candidates

spec = load_spectrum("data/XPS_Si2p_siloxane_synthetic.csv").crop(98.0, 104.5)
pool = [
    Component.from_line("Au4f", "Au0", name="Au4f_ghost", center=99.8, center_sigma=0.5),
    Component.from_line("Si2p", "Si0"),
    Component.from_line("Si2p", "SiOx"),
    Component.from_line("Si2p", "SiO2"),
]
sel = select_model(spec, subset_candidates(pool, min_size=2), background="auto")
print(sel.best.peak_table())              # 中心・FWHM・面積・面積%
print(sel.n_component_probabilities())    # 成分数ごとの近似事後確率
```

## ベンチマーク: Auゴースト上のSi 2p

開発の基準にしたのは意図的に選んだ難スペクトルです（Au電極上の薄いシロキサン層。Si 2p領域がAuゴースト線と重なり、SiO2ショルダーは急峻な背景上の微弱構造）。物理的に正しい分解（SiO2 102–103 eV / SiOx≒101 eV / ゴーストまたは元素状Si 99–100 eV）がBICで自動選択されます:

| 成分 | 中心 (eV) | 面積% |
|------|-----------|-------|
| Si0 / Auゴースト | 99.91 | 70.0 |
| SiOx（シロキサン） | 101.43 | 16.1 |
| SiO2 | 102.73 | 13.9 |

実測定データは所有権の関係で配布していません。代わりに、上記のフィットモデルから実測相当ノイズで生成した**合成等価データ** `data/XPS_Si2p_siloxane_synthetic.csv` を同梱しています。SiO2ショルダーが検出限界近傍（成分数事後確率 n=3: 60% / n=2: 40%）という実測定の難しさもそのまま再現しており、回帰テスト（`tests/test_benchmark_ng.py`）として固定されています。

## ケーススタディ: GaN初期酸化のオペランド追跡（Spring-8 BL23SU）

`data2/` に同梱の公開データセット（c面GaN初期酸化のO 1s系列16本、hv=730 eV、NIMS MDR登録データ）を、化学状態プール（Ga-O / OH / H₂O）×背景のBIC自動選択で**全16本・約6秒・人手ゼロ**で分解できます:

```bash
python -X utf8 benchmarks/casestudy_gan_oxidation.py
```

Ga-O成分の急速飽和とOH成分の継続成長という酸化カイネティクス、およびGa-O中心の化学シフト発展が自動抽出されます（`benchmarks/figures/gan_kinetics.png`）。運動エネルギー軸のデータは `load_spectrum(path, hv=730.0)` / CLI `--hv 730` でBEに変換されます。

## 他手法との違い

XPSスペクトルの自動解析には優れた先行研究があり、本パッケージはそれらから多くを学んでいます。**以下は優劣の比較ではなく、設計目標の違いの整理です**。用途によって適切なツールは異なります。

| | EMPeaks（松村・安藤ら） | 篠塚らのベイズ法（NIMS） | xps-peakfit（本パッケージ） |
|---|---|---|---|
| 主目的 | 多数スペクトル系列の高速ピークシフト追跡 | 単一スペクトルの厳密なベイズ推定・不確かさ定量化 | 単一難スペクトルの化学状態分解の実用自動化 |
| 最適化 | スペクトル適応EM/ECM（微分不要） | レプリカ交換モンテカルロ | lmfit最小二乗＋適応マルチスタート |
| ピーク本数 | ユーザー指定（K固定） | ベイズ自由エネルギーF(K)で自動 | BIC＋近似事後確率で自動 |
| 背景 | 混合成分（uniform/linear/ramp） | 両端強度パラメータ＋事前分布 | Active Shirley/Tougaard（BICで自動選択） |
| 物理知識 | 意図的に不使用（データ駆動） | スピン軌道定数のみタイト事前分布 | ラインDB（分裂・比率・化学シフト事前分布・幅共有） |
| 不確かさ | — | 事後分布から厳密に | opt-in emcee MCMC（数十秒） |
| 得意な場面 | オペランド計測などの大量系列 | 論文グレードの厳密解析 | 金属基板・ゴースト重畳などの難単一スペクトル |

- **EMPeaks**（産総研 松村太郎次郎氏・東京科学大 安藤康伸氏ら）: スペクトル強度を混合分布の重みとみなす独創的なEM定式化で、事前知識なしのデータ駆動解析と高スループット処理を実現しています。`benchmarks/` に同一データでの参照比較を置いていますが、これは**EMPeaksがpipで入手できる貴重な公開実装であるため参照点として使用したもの**であり、想定用途が異なる両者の優劣を競うものではありません（詳細な公平性の注記は `benchmarks/RESULTS.md` を参照）。
- **篠塚寛志氏ら（NIMS）のベイズ推定法**: レプリカ交換MCによる完全ベイズでピーク数の事後確率や信頼区間まで得られる、方法論的に最も厳密なアプローチです。本パッケージの「スピン軌道定数のタイト事前分布」（`vary_so`）と「成分数事後確率の表示」はこの研究系譜から着想を得ています。
- **本パッケージの立ち位置**: 両者の中間です。物理知識（ラインDB）を明示的に注入することで、1スペクトルあたり数秒の実用速度のまま、「SiO2が13.9%」という化学的に解釈済みの答えと、その信頼区間（opt-in）を返すことを狙っています。

## ラインデータベース

`xps_peakfit/lines.py` にスピン軌道定数と化学状態事前分布を登録（Si2p, Au4f, C1s, O1s, Cr2p, Ni2p同梱。`register(LineShape(...))` で拡張可能）。ゴースト線は登録済みラインの形状を借りて位置事前分布だけ自由化する仕組みで、フィッティングエンジンに元素固有コードは一切ありません。

## テスト

```bash
pytest            # 42テスト（実データベンチマーク6件を含む）
pytest -m "not slow"
```

## ライセンス

MIT
