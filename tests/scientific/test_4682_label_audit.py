from preanchor.labels.validation import validate_label_rows
def test_primary_fallback_schema():
    row={k:0 for k in ('importance_released','importance_legacy','importance_canonical','importance_primary')}
    row['label_source']='canonical_fallback'
    assert validate_label_rows([row],primary_source='canonical_fallback')['rows']==1
