# PROSODIC MARKOV CHAIN: staged next-word prediction and GIF
# Run in RStudio with Source. Required once: install.packages("gifski")
# No neural network, embeddings, sentence meaning, or future audio inputs.
#
# State X[t] = (word[t], pitch category[t]). First-order assumption:
# P(X[t+1] | X[1], ..., X[t]) = P(X[t+1] | X[t]).
# Fit joint-state transitions by maximum likelihood:
# A[i,j] = count(i -> j) / sum_j count(i -> j).
# P(next word = v | X[t]) = sum_{j: word(j)=v} A[X[t],j].
# Earlier history matters only through the latest state in this version.
# Stages below are visualization steps, not hidden psychological states.
#
# DEMO LIMIT: training prosody/continuation associations are deliberately
# simulated. They are not established English speech rules or evidence of
# human comprehension. A separate trial is replayed without updating counts.
# Real-data validity needs held-out speakers, acoustic audits and behavior.
# GIF API: https://cran.r-project.org/web/packages/gifski/refman/gifski.html

# ----------------------- 1. SETTINGS ------------------------------------
MODE <- "demo"                 # "demo" or "csv"
TRAIN_CSV <- "training_words.csv"
TRIAL_CSV <- "trial_words.csv"  # exactly one separate utterance
OUTPUT_DIR <- file.path(getwd(), "prosodic_markov_output")
PITCH_THRESHOLD <- 1            # semitones; illustrative fixed threshold
FRAME_SECONDS <- 1.4
WIDTH <- 1200L
HEIGHT <- 850L
set.seed(42)

# No automatic package installation or downloads.
if (!requireNamespace("gifski", quietly = TRUE)) {
  stop('Run install.packages("gifski") once, then Source this script again.')
}
stopifnot(MODE %in% c("demo", "csv"), PITCH_THRESHOLD > 0)
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ----------------------- 2. WORD MEASUREMENTS ---------------------------
# CSV schema: utterance_id, word_index, word, pitch_change_st
# pitch_change_st = 12 * log2(end_F0 / start_F0), using valid voiced
# measurements WITHIN the completed word. NA means unavailable pitch.
# Split utterances at speaker changes. Do not join across those boundaries.
# Obtain word boundaries and measurements externally; this is not ASR.
# Audit extraction so smoothing/windows do not use audio after the word.
# Full-transcript alignment is an oracle-aligned offline condition.

pitch_state <- function(x) {
  ifelse(!is.finite(x), "unavailable",
         ifelse(x > PITCH_THRESHOLD, "rising",
                ifelse(x < -PITCH_THRESHOLD, "falling", "level")))
}

make_utterance <- function(id, words, pitch) {
  data.frame(utterance_id = id, word_index = seq_along(words),
             word = words, pitch_change_st = pitch,
             stringsAsFactors = FALSE)
}

if (MODE == "demo") {
  # Synthetic association: after "some", rising favors tea and falling
  # favors coffee. This artificial rule makes the mechanism visible.
  examples <- lapply(seq_len(360), function(i) {
    rising <- runif(1) < 0.5
    drink <- sample(c("tea", "coffee", "water"), 1,
                    prob = if (rising) c(.70, .15, .15) else c(.15, .70, .15))
    words <- c("i", sample(c("would", "might"), 1), "like", "some",
               drink, sample(c("please", "thanks"), 1))
    pitch <- rnorm(6, 0, .25)
    pitch[4] <- if (rising) 2.5 else -2.5
    pitch[6] <- -2.5
    make_utterance(sprintf("train_%03d", i), words, pitch)
  })
  training <- do.call(rbind, examples)
  trial <- make_utterance("trial_only",
                         c("i", "would", "like", "some", "tea", "please"),
                         c(0, .2, -.3, 2.5, .1, -2.5))
} else {
  training <- read.csv(TRAIN_CSV, stringsAsFactors = FALSE)
  trial <- read.csv(TRIAL_CSV, stringsAsFactors = FALSE)
}

prepare <- function(d) {
  required <- c("utterance_id", "word_index", "word", "pitch_change_st")
  if (!all(required %in% names(d))) stop("Missing CSV columns: ",
                                          paste(setdiff(required, names(d)), collapse = ", "))
  stopifnot(nrow(d) > 0, is.numeric(d$word_index),
            is.numeric(d$pitch_change_st), !anyNA(d$utterance_id),
            !anyNA(d$word), !anyNA(d$word_index))
  d$word <- tolower(trimws(d$word)) # preserves contractions/disfluencies
  stopifnot(all(nzchar(d$word)), all(nzchar(as.character(d$utterance_id))),
            !any(grepl("|", d$word, fixed = TRUE)),
            !any(d$word %in% c("<end>")))
  d <- d[order(d$utterance_id, d$word_index), required]
  groups <- split(d, d$utterance_id)
  for (g in groups) {
    if (!identical(as.integer(g$word_index), seq_len(nrow(g))) ||
        any(g$word_index != as.integer(g$word_index))) {
      stop("word_index must be consecutive integers from 1 within each utterance.")
    }
  }
  d$prosody <- pitch_state(d$pitch_change_st)
  d$state <- paste(d$word, d$prosody, sep = "|")
  rownames(d) <- NULL
  d
}
training <- prepare(training)
trial <- prepare(trial)
stopifnot(length(unique(trial$utterance_id)) == 1L)
if (any(trial$utterance_id %in% training$utterance_id)) {
  stop("Use distinct training and trial utterance IDs.")
}

# ----------------------- 3. FIT MARKOV CHAIN ----------------------------
fit_chain <- function(d) {
  transitions <- do.call(rbind, lapply(split(d, d$utterance_id), function(g) {
    data.frame(from = g$state, to = c(g$state[-1], "<end>|terminal"))
  }))
  states <- sort(unique(c(transitions$from, transitions$to)))
  counts <- as.matrix(table(factor(transitions$from, levels = states),
                            factor(transitions$to, levels = states)))
  dimnames(counts) <- list(states, states)
  # END is absorbing by definition, not an estimated observed transition.
  terminal <- match("<end>|terminal", states)
  totals <- rowSums(counts)
  A <- matrix(0, length(states), length(states), dimnames = dimnames(counts))
  observed <- totals > 0
  A[observed, ] <- counts[observed, , drop = FALSE] / totals[observed]
  A[terminal, terminal] <- 1
  state_words <- sub("\\|.*$", "", states)
  list(counts = counts, A = A, states = states, state_words = state_words,
       totals = totals, transitions = transitions)
}
model <- fit_chain(training)
stopifnot(all(abs(rowSums(model$A) - 1) < 1e-10))

predict_next <- function(word, prosody, model) {
  # Only these two observed values enter prediction. No trial target input.
  state <- paste(word, prosody, sep = "|")
  if (!state %in% model$states || model$totals[state] == 0) {
    stop("Unseen state: ", state,
         ". Raw MLE is undefined here; supply training examples or explicitly add smoothing.")
  }
  p <- tapply(model$A[state, ], model$state_words, sum)
  counts <- tapply(model$counts[state, ], model$state_words, sum)
  ans <- data.frame(word = names(p), probability = as.numeric(p),
                    count = as.numeric(counts), stringsAsFactors = FALSE)
  ans <- ans[order(-ans$probability, ans$word), ]
  stopifnot(abs(sum(ans$probability) - 1) < 1e-10)
  ans
}

records <- lapply(seq_len(nrow(trial)), function(t) {
  p <- predict_next(trial$word[t], trial$prosody[t], model)
  # Target accessed only AFTER prediction, for reveal and scoring.
  actual <- if (t < nrow(trial)) trial$word[t + 1] else "<end>"
  q <- p$probability[match(actual, p$word)]
  if (is.na(q)) q <- 0
  list(t = t, prediction = p, actual = actual, probability_actual = q,
       surprisal_bits = if (q > 0) -log2(q) else Inf)
})
# Zero probability for unseen transitions is intentional raw MLE behavior.
# Infinite surprisal is retained, never silently clipped or called smoothing.

# Sanity check of the deliberately simulated association.
if (MODE == "demo") {
  check <- predict_next("some", "rising", model)
  stopifnot(check$word[1] == "tea")
  stopifnot(predict_next("some", "falling", model)$word[1] == "coffee")
}

# ----------------------- 4. DRAW FOUR STAGES ----------------------------
stage_names <- c("OBSERVE", "ENCODE STATE", "PREDICT", "REVEAL")
blue <- "#246B8E"; orange <- "#CB702C"; green <- "#26826C"
draw_frame <- function(rec, stage) {
  t <- rec$t
  par(mfrow = c(2, 2), mar = c(4, 4, 3, 1), oma = c(2, 0, 5, 0),
      bg = "#FAFAF7", fg = "#253341", col.axis = "#253341")

  # Panel 1: only the observed prefix, not the future sentence.
  plot.new(); plot.window(xlim = c(0, 1), ylim = c(0, 1))
  text(0, .97, "1 / Speech observed so far", adj = 0, font = 2, cex = 1.25)
  prefix <- paste(trial$word[seq_len(t)], collapse = " ")
  text(0, .74, paste(strwrap(prefix, width = 42), collapse = "\n"),
       adj = c(0, 1), cex = 1.5, col = blue)
  text(0, .37, paste("Completed word:", trial$word[t]), adj = 0, cex = 1.1)
  delta <- trial$pitch_change_st[t]
  text(0, .25, if (is.finite(delta)) sprintf("Pitch change: %+.2f semitones", delta)
       else "Pitch unavailable", adj = 0)
  text(0, .10, "Only completed-word information is available.", adj = 0, cex = .9)

  # Panel 2: observed pitch changes, not invented raw audio contours.
  x <- seq_len(t); y <- trial$pitch_change_st[x]
  lim <- max(3, abs(y[is.finite(y)]), na.rm = TRUE)
  plot(x, y, type = "b", pch = 19, col = blue,
       xlim = c(.5, nrow(trial) + .5), ylim = c(-lim, lim), xaxt = "n",
       xlab = "Completed words", ylab = "Pitch change (semitones)",
       main = "2 / Observed prosodic variation")
  axis(1, at = x, labels = trial$word[x])
  abline(h = c(-PITCH_THRESHOLD, PITCH_THRESHOLD), lty = 3, col = "grey60")
  abline(h = 0, col = "grey85")

  # Panel 3: outgoing word probabilities, marginalized over future prosody.
  plot.new(); plot.window(xlim = c(0, 1), ylim = c(0, 1))
  text(0, .97, "3 / Markov transition", adj = 0, font = 2, cex = 1.25)
  if (stage >= 2) {
    rect(.01, .38, .42, .67, col = "#E1EDF2", border = blue)
    text(.215, .525, paste(trial$word[t], trial$prosody[t], sep = "\n"), cex = 1.2)
    if (stage >= 3) {
      top <- head(rec$prediction[rec$prediction$probability > 0, ], 4)
      ys <- seq(.77, .17, length.out = nrow(top))
      for (j in seq_len(nrow(top))) {
        arrows(.43, .525, .69, ys[j], length = .08, col = blue)
        text(.56, (ys[j] + .525) / 2 + .035,
             sprintf("%.2f", top$probability[j]), cex = .9)
        text(.73, ys[j], top$word[j], adj = 0, col = blue, cex = 1.15)
      }
    }
    text(0, .03, "State = current word + current pitch category", adj = 0, cex = .9)
  } else text(.5, .5, "Waiting to encode the observed word...", cex = 1.05)

  # Panel 4: probability bars, then the observed outcome.
  if (stage >= 3) {
    top <- head(rec$prediction[rec$prediction$probability > 0, ], 5)
    cols <- rep(blue, nrow(top))
    if (stage == 4) cols[top$word == rec$actual] <- green
    bp <- barplot(top$probability, names.arg = top$word, ylim = c(0, 1.18),
                  col = cols, border = NA, ylab = "Next-word probability",
                  main = "4 / Count-based maximum likelihood")
    labels <- sprintf("%d/%d = %.2f", top$count,
                      model$totals[trial$state[t]], top$probability)
    text(bp, top$probability + .06, labels, cex = .85)
    if (stage == 4) {
      mtext(sprintf("Actual: %s | P = %.3f | Surprisal = %.2f bits",
                    rec$actual, rec$probability_actual, rec$surprisal_bits),
            side = 1, line = 2.6, cex = .85, col = green)
    }
  } else {
    plot.new(); text(.5, .55, "Probabilities appear during PREDICT.", cex = 1.1)
  }
  mtext("PROSODIC WORD PREDICTION / FIRST-ORDER MARKOV CHAIN", outer = TRUE,
        side = 3, line = 2.9, font = 2, cex = 1.25)
  mtext(sprintf("Word %d of %d  |  Stage %d: %s", t, nrow(trial), stage,
                stage_names[stage]), outer = TRUE, side = 3, line = 1.2, col = orange)
  mtext(if (MODE == "demo") "SIMULATED SPEECH / Illustrates mechanics, not human-comprehension accuracy"
        else "OFFLINE ALIGNED WORD INPUT / Human-comprehension validity is not established",
        outer = TRUE, side = 1, line = .5, cex = .9)
}

# ----------------------- 5. EXPORT TABLES AND GIF -----------------------
write.csv(model$counts, file.path(OUTPUT_DIR, "transition_counts.csv"))
write.csv(model$A, file.path(OUTPUT_DIR, "transition_matrix.csv"))
write.csv(training, file.path(OUTPUT_DIR, "training_words_used.csv"), row.names = FALSE)
write.csv(trial, file.path(OUTPUT_DIR, "trial_words_used.csv"), row.names = FALSE)
scores <- do.call(rbind, lapply(records, function(r) {
  data.frame(step = r$t, current_word = trial$word[r$t], prosody = trial$prosody[r$t],
             actual_next = r$actual, probability = r$probability_actual,
             surprisal_bits = r$surprisal_bits)
}))
write.csv(scores, file.path(OUTPUT_DIR, "trial_predictions.csv"), row.names = FALSE)

# Use a new temporary folder each run so old frames never enter the GIF.
frames_dir <- tempfile("markov_frames_")
dir.create(frames_dir)
frames <- character()
render_png <- function(path, rec, stage) {
  png(path, width = WIDTH, height = HEIGHT, res = 110)
  on.exit(dev.off(), add = TRUE)
  draw_frame(rec, stage)
}
for (r in records) {
  for (stage in seq_along(stage_names)) {
    path <- file.path(frames_dir, sprintf("frame_%03d.png", length(frames) + 1L))
    render_png(path, r, stage)
    frames <- c(frames, path)
  }
}
gif_path <- file.path(OUTPUT_DIR, "prosodic_markov_stages.gif")
gifski::gifski(png_files = frames, gif_file = gif_path,
               width = WIDTH, height = HEIGHT, delay = FRAME_SECONDS, loop = TRUE)
stopifnot(file.exists(gif_path), file.info(gif_path)$size > 0)
cat("\nGIF saved to:\n", normalizePath(gif_path, winslash = "/"), "\n\n")
print(scores)

# Display the completed GIF in RStudio's Viewer.
htmltools::html_print(
  htmltools::tags$img(
    src = base64enc::dataURI(file = gif_path, mime = "image/gif"),
    style = "width:100%;"
  ),
  viewer = rstudioapi::viewer
)
# Optional: open OUTPUT_DIR in your file browser and double-click the GIF.
# Model remains fixed throughout replay. The revealed word supplies the next
# observed state; it is not used retroactively in the preceding prediction.

# Select transitions leaving the word "some".
transitions <- model$transitions
selected <- transitions[
  transitions$from %in% c("some|rising", "some|falling"),
  ,
  drop = FALSE
]

# Extract current pitch category and next word.
selected$pitch <- sub("^some\\|", "", selected$from)
selected$next_word <- sub("\\|.*$", "", selected$to)

# Count combinations.
counts <- table(selected$pitch, selected$next_word)
print(counts)

# Show next-word proportions within each pitch category.
print(round(prop.table(counts, margin = 1), 3))

# Test association.
result <- chisq.test(counts)
print(result)

# Inspect expected counts: small counts can weaken the approximation.
print(result$expected)

# Cramer's V: strength of association.
cramers_v <- sqrt(
  as.numeric(result$statistic) /
    (sum(counts) * min(nrow(counts) - 1, ncol(counts) - 1))
)
print(cramers_v)
