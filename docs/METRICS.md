# 評価指標

[日本語](#japanese) | [English](#english)

<a id="japanese"></a>

この文書では、**常用漢字読みベンチマーク Parakeet Edition**の評価指標と、その計算方法について説明します。

## 指標の一覧

| 指標 | 内容 |
|---|---|
| Accuracy | 評価対象の読みが自然な読みと完全に一致した割合 |
| Relaxed Accuracy | 評価対象の読みが自然な読みまたは許容可能な読みと完全に一致した割合 |
| Target Kana-CER | 評価対象の漢字の読みを、自然な読みと比較したカナCER |
| Target Kana-CER@1 | 例文ごとのTarget Kana-CERを最大`1.0`に制限した値 |
| Relaxed Target Kana-CER | 評価対象の漢字の読みを、自然な読みまたは許容可能な読みと比較したカナCER |
| Relaxed Target Kana-CER@1 | 例文ごとのRelaxed Target Kana-CERを最大`1.0`に制限した値 |
| Sentence Kana-CER | 文全体のカナ列に対するCER |
| Sentence Kana-CER@1 | 例文ごとのSentence Kana-CERを最大`1.0`に制限した値 |
| Text CER | TTS音声の通常の漢字仮名交じり文での書き起こしと元の文とのCER |
| Text CER@1 | 例文ごとのText CERを最大`1.0`に制限した値 |

元のJoyo Kanji Yomi Benchmarkにおける`Kana-CER`はTarget Kana-CERに、`Sent-Kana-CER`はSentence Kana-CERに相当します。ただし、データ、正規化、およびアライメント手法が異なるため、数値に互換性はありません。

Accuracy、Relaxed Accuracy、Target Kana-CER、Target Kana-CER@1、Relaxed Target Kana-CER、Relaxed Target Kana-CER@1については、全体値と併せて`reading_category`ごとの値も出力します。分類は音読み（`on_yomi`）、訓読み（`kun_yomi`）、常用漢字表の付表の語（`joyo_appendix_reading`）の3つです。

## 指標の詳細について

以下では、指標の計算の詳細について述べます。

### CERとCER@1

**Kana-CER**は、通常のCER (Character Error Rate) をカタカナ列に対して適用したものです。
**Kana-CER@1**は、Kana-CERの例文ごとのCERを1.0 (= 100%) でクリップした値です。
上記のコーパス全体の値は、各例文のCERまたはCER@1を平均したものです。

### 音声からの読みの取得

TTSの評価では、音声からカナ列を得る必要があります。本ベンチマークでは、オリジナルと同様、ASRモデルとして[`sbintuitions/kana-whisper`](https://huggingface.co/sbintuitions/kana-whisper)を使用して、カナ列への書き起こしを行います。

### 読みの正規化

`学校`に対する`ガッコウ`と`ガッコー`のような読み表記の違いを吸収するため、Kana-CERを計算する前に、正解と予測結果の双方に次の正規化を適用します。

- NFKCで正規化する
- 句読点、記号、および空白を除去する
- ひらがなをカタカナへ変換する
- `ヂ/ヅ/ヲ/ヰ/ヱ`を`ジ/ズ/オ/イ/エ`へ統一する
- 長音符を直前の仮名の母音へ変換する
- オ段の仮名に続く`ウ`を`オ`へ変換する
- エ段の仮名に続く`イ`を`エ`へ変換する


| 表記 | 正規化後 |
|---|---|---|
| `ガッコウ` | `ガッコオ` |
| `ガッコー` | `ガッコオ` |
| `ケイエイ` | `ケエエエ` |
| `ケーエー` | `ケエエエ` |

注意: 助詞「は」「へ」のデータ上での読みの正解は「ワ」「エ」であり、正規化の過程でも`ハ`と`ワ`、`ヘ`と`エ`は別のものとして扱います。

### Target Kana-CERとRelaxed Target Kana-CER

Target Kana-CERは、評価対象の漢字の読みに対応する部分だけを比較したKana-CERです。

例：「天気は<晴>れ」の読みに対して、

```text
正解: テンキワ<ハ>レ
予測結果: テンキワセイレ
```

となったとします。このとき次のプロセスで計算します。

- `readings.natural`の各候補について、前後の文字列を使って対象範囲を個別にアライメントする
- 前後の文字列との編集距離を最優先し、同点の場合はその候補との編集距離、元のタグ位置からの移動量の順で対象範囲を決める
- 各候補と、その候補用に取り出した読みとのCERを計算し、最も小さい値をTarget Kana-CERとする
- Relaxed Target Kana-CERでは、`readings.natural`に加えて`readings.marginal`の各候補も同じ方法で個別にアライメントし、その中で最も小さいCERを使用する

候補ごとにアライメントするため、読みの長さや語中の分割位置が異なる候補も正しく評価できます。例えば、`<アメ>ツチ`に対する予測`テンチ`では、`テン`を対象漢字の許容可能な読みとして評価できます。この場合、タグ外の`地`に対応する`ツチ/チ`の差はTarget Kana-CERには含めず、Sentence Kana-CERに反映します。文頭・文末に前後の文字列がない場合、その側の対象境界は文頭・文末に固定します。したがって、文中の無関係な位置に同じ読みがあるだけでは正解になりません。

`details/all.jsonl`では、`mapped_target`および`alignment_*`が最良のnatural候補のアライメントを、`relaxed_mapped_target`および`relaxed_alignment_*`がnaturalとmarginalを合わせた最良候補のアライメントを記録します。

### Accuracy

**Accuracy**は、JKYB-Parakeetデータセットの全ての文のうち、予測結果から取り出した評価対象の漢字の読みに対応するカナ列が、`readings.natural`のいずれかと完全に一致した例文の割合です。

**Relaxed Accuracy**は、条件を緩めて、全ての文のうち、予測された読みが`readings.natural`または`readings.marginal`のいずれかと完全に一致した例文の割合です。

### Sentence Kana-CER

Sentence Kana-CERは、データの`yomi`と、予測された文全体の読みのCERです。

評価対象以外の箇所には、`寂しい`の`サビシイ/サミシイ`、`日本`の`ニホン/ニッポン`のように、複数の自然な読みが存在する場合があり、この評価指標はそれを考慮していないことに注意してください。

### Text CER

音声に対して[`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo)で漢字仮名交じり文での書き起こしを行い、結果をデータの`text`と比較したCERです。

### 出力がない例文

`--allow-missing`を指定した場合、G2Pの予測結果やTTSの音声がない例文も評価から除外しません。空の出力として扱い、その例文のTarget Kana-CER、Relaxed Target Kana-CER、Sentence Kana-CER、およびText CERを`1.0`、AccuracyとRelaxed Accuracyを不正解として集計します。評価結果には入力の件数と不足していた例文も記録されます。

---

<a id="english"></a>

# Metrics

This document explains the metrics and their calculation methods for the **Joyo Kanji Yomi Benchmark: Parakeet Edition**.

## Metric Overview

| Metric | Description |
|---|---|
| Accuracy | Percentage of evaluated readings that exactly match a natural reading |
| Relaxed Accuracy | Percentage that exactly match either a natural or a marginal reading |
| Target Kana-CER | Kana-CER against the natural readings of the evaluated kanji |
| Target Kana-CER@1 | Target Kana-CER clipped to a maximum of `1.0` for each example |
| Relaxed Target Kana-CER | Kana-CER against the natural or marginal readings of the evaluated kanji |
| Relaxed Target Kana-CER@1 | Relaxed Target Kana-CER clipped to a maximum of `1.0` for each example |
| Sentence Kana-CER | CER for the full-sentence kana sequence |
| Sentence Kana-CER@1 | Sentence Kana-CER clipped to a maximum of `1.0` for each example |
| Text CER | CER between a standard Japanese transcription of TTS audio in kanji and kana and the source sentence |
| Text CER@1 | Text CER clipped to a maximum of `1.0` for each example |

The original Joyo Kanji Yomi Benchmark's `Kana-CER` corresponds to Target Kana-CER, while `Sent-Kana-CER` corresponds to Sentence Kana-CER. The scores are not numerically compatible because the data, normalization, and alignment methods differ.

Accuracy, Relaxed Accuracy, Target Kana-CER, Target Kana-CER@1, Relaxed Target Kana-CER, and Relaxed Target Kana-CER@1 are also reported by `reading_category`. The three categories are On’yomi (`on_yomi`), Kun’yomi (`kun_yomi`), and words from the Jōyō Kanji Table appendix (`joyo_appendix_reading`).

## Metric Details

The following sections describe how each metric is calculated.

### CER and CER@1

**Kana-CER** applies standard CER (Character Error Rate) to katakana sequences.
**Kana-CER@1** clips the CER of each example to `1.0` (= 100%).
Corpus-level values are calculated by averaging the CER or CER@1 of all examples.

### Obtaining Readings from Audio

TTS evaluation requires a kana sequence to be obtained from each audio file. As in the original benchmark, JKYB-Parakeet uses the [`sbintuitions/kana-whisper`](https://huggingface.co/sbintuitions/kana-whisper) ASR model to transcribe the audio into kana.

### Reading Normalization

Before calculating Kana-CER, the reference and prediction are both normalized as follows so that notational differences such as `ガッコウ` and `ガッコー` are not counted as errors.

- Apply NFKC normalization
- Remove punctuation, symbols, and whitespace
- Convert hiragana to katakana
- Fold `ヂ/ヅ/ヲ/ヰ/ヱ` into `ジ/ズ/オ/イ/エ`
- Replace a prolonged sound mark with the vowel of the preceding kana
- Replace `ウ` after an o-row kana with `オ`
- Replace `イ` after an e-row kana with `エ`

| Form | Normalized |
|---|---|
| `ガッコウ` | `ガッコオ` |
| `ガッコー` | `ガッコオ` |
| `ケイエイ` | `ケエエエ` |
| `ケーエー` | `ケエエエ` |

Note: The correct readings of the particles `は` and `へ` in the data are `ワ` and `エ`. The normalization process continues to distinguish `ハ` from `ワ` and `ヘ` from `エ`.

### Target Kana-CER and Relaxed Target Kana-CER

Target Kana-CER compares only the part corresponding to the evaluated kanji.

For example, suppose the reading of `天気は<晴>れ` has the following reference and prediction:

```text
Reference:  テンキワ<ハ>レ
Prediction: テンキワセイレ
```

The metric is calculated through the following process:

- Align the target span independently for every candidate in `readings.natural`, using the surrounding strings as context
- Prefer the lowest context edit distance; break ties by the edit distance to that candidate and then by displacement from the original tagged boundary
- Compare each candidate with the span extracted for that candidate and use the smallest CER as the example's Target Kana-CER
- For Relaxed Target Kana-CER, independently align every candidate in both `readings.natural` and `readings.marginal`, then use the smallest CER

Candidate-specific alignment supports alternatives with different lengths or different divisions inside a word. For example, given reference `<アメ>ツチ` and prediction `テンチ`, `テン` can be evaluated as an acceptable reading of the target kanji. The `ツチ/チ` difference outside the tag does not affect Target Kana-CER; it remains visible in Sentence Kana-CER. When one side has no surrounding context because the target is at the beginning or end of the sentence, that boundary is anchored to the corresponding sentence edge. An identical reading at an unrelated position therefore cannot rescue the target.

In `details/all.jsonl`, `mapped_target` and the `alignment_*` fields describe the best natural candidate alignment. `relaxed_mapped_target` and the `relaxed_alignment_*` fields describe the best alignment across both natural and marginal candidates.

### Accuracy

**Accuracy** is the percentage of all examples in JKYB-Parakeet whose extracted kana sequence for the evaluated kanji exactly matches one of the readings in `readings.natural`.

**Relaxed Accuracy** uses a less strict condition: it is the percentage of all examples whose predicted reading exactly matches a reading in either `readings.natural` or `readings.marginal`.

### Sentence Kana-CER

Sentence Kana-CER is the CER between the dataset `yomi` and the predicted full-sentence reading.

Note that parts outside the evaluated kanji can have multiple natural readings, such as `サビシイ/サミシイ` for `寂しい` and `ニホン/ニッポン` for `日本`, and this metric does not account for them.

### Text CER

Text CER is calculated by transcribing the audio into conventional Japanese text with kanji and kana using [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo), and comparing the result with the dataset `text`.

### Examples Without Output

When `--allow-missing` is specified, examples without G2P predictions or TTS audio are not excluded. They are treated as empty outputs: Target Kana-CER, Relaxed Target Kana-CER, Sentence Kana-CER, and Text CER are scored as `1.0`, while Accuracy and Relaxed Accuracy count them as incorrect. The report records both input coverage and the affected examples.
