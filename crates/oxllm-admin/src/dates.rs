//! Hand-rolled civil-calendar helpers (Howard Hinnant's algorithms) for
//! `YYYY-MM-DD` dates in UTC. Kept dependency-free; swap for `jiff` if it
//! ever becomes a dependency of this crate.

/// Calendar date `YYYY-MM-DD` in UTC.
pub(crate) fn today_iso_date() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    civil_date_from_unix_days(secs.div_euclid(86_400))
}

pub(crate) fn days_before_iso(iso: &str, days: i64) -> Option<String> {
    let day_num = parse_iso_date_to_unix_days(iso)?;
    Some(civil_date_from_unix_days(day_num - days))
}

pub(crate) fn parse_iso_date_to_unix_days(iso: &str) -> Option<i64> {
    let parts: Vec<&str> = iso.split('-').collect();
    if parts.len() != 3 {
        return None;
    }
    let y: i32 = parts[0].parse().ok()?;
    let m: i32 = parts[1].parse().ok()?;
    let d: i32 = parts[2].parse().ok()?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    // Days since Unix epoch (1970-01-01) via civil algorithm (Howard Hinnant).
    let y_adj = if m <= 2 { y - 1 } else { y };
    let era = y_adj.div_euclid(400);
    let yoe = y_adj - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let day_num = i64::from(era) * 146_097 + i64::from(doe) - 719_468;
    // Round-trip check: an out-of-range day-of-month (e.g. 2024-02-30) maps to
    // a valid day count that the inverse algorithm shifts into the next month.
    // Reject such inputs by comparing the reconstructed date against the input.
    let rt = civil_date_from_unix_days(day_num);
    let rp: Vec<&str> = rt.split('-').collect();
    if rp.len() != 3
        || rp[0].parse::<i32>().ok() != Some(y)
        || rp[1].parse::<i32>().ok() != Some(m)
        || rp[2].parse::<i32>().ok() != Some(d)
    {
        return None;
    }
    Some(day_num)
}

pub(crate) fn civil_date_from_unix_days(z: i64) -> String {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i32 + era as i32 * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}
