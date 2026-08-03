# Shortening a chart does not advance the label time

Owner-reviewed boxes describe a completed visual pattern. Reducing a chart from 200 to 96 bars gives the
same narrow pattern roughly twice as many horizontal pixels, but it does not make that pattern complete at an
earlier candle.

The safe short-window experiment therefore separates two facts:

- `box_end_time` is the last candle covered by the owner box;
- `available_at` is the close of the last candle rendered into the model input.

To avoid the old fixed-right-edge shortcut, short examples use a deterministic 0/8/16/24-bar right context.
That creates several real box positions without flips, translations, mosaic, or synthetic future bars. The
zero-context quarter represents the earliest complete-box view; later contexts are position-invariance
examples, not claims that the signal was available at the earlier box time.

An actually earlier detector needs a separately validated partial-pattern target. It must not be fabricated by
silently moving the completed owner box to an earlier candle.
