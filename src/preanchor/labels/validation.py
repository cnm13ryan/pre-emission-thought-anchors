"""Structural validation for Stage 4 label tables."""
def validate_label_rows(rows: list[dict], *, primary_source: str) -> dict:
    if primary_source not in {'released','canonical_fallback'}: raise ValueError('invalid primary source')
    required={'importance_released','importance_legacy','importance_canonical','importance_primary','label_source'}
    for row in rows:
        if not required <= row.keys(): raise ValueError('label schema missing fields')
        if row['label_source'] != primary_source: raise ValueError('mixed primary sources')
        if row['importance_primary'] is None or row['importance_primary'] < 0: raise ValueError('invalid primary label')
    return {'schema_version':1,'rows':len(rows),'primary_source':primary_source}
