use tracing_subscriber::fmt::format::FmtSpan;

pub fn init_tracing() {
    // stderr, not the default stdout: a tool like face-hint prints its actual
    // result — clean JSON, meant to be piped straight into another program —
    // on stdout, and a log line mixed into that breaks every parser reading it.
    tracing_subscriber::fmt()
        .with_span_events(FmtSpan::CLOSE)
        .with_writer(std::io::stderr)
        .init();
}
