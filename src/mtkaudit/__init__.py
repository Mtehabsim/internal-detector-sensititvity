"""mtkaudit: code for "High AUROC, Low Detection: Calibration and Reference Sensitivity in Model-Internal
Jailbreak Detectors".

Modules
    paths       repository root and environment configuration
    release     harness around the MTK release (pinned copy, sampling, text-disjoint calibration, scoring)
    mechanism   interleaving counts behind MTK's rank feature
    matched     the matched prompt set every detector is scored on
    gradsafe    GradSafe: official code (official) and the chat-template port (port)
    baselines   HiddenDetect (LLM adaptation), windowed perplexity, model loading
    stats       bootstrap helpers
    results     loaders for the archived results under results/
    caches      exact comparison of feature caches
    cli         command-line helpers (--help for option-less scripts, absolute path options)
"""
