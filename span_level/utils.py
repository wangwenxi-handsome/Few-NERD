def get_span_pos(start, end, sen_len, max_span_length = None):
    assert end >= start
    if max_span_length is None:
        past = int(start * sen_len - start * (start - 1) / 2)
        return past + (end - start)
    else:
        assert end - start < max_span_length
        raise ValueError


def get_span_start_end(pos, sen_len):
    num = 0
    for i in range(len(sen_len) - 1):
        for j in range(i, len(sen_len)):
            if num == pos:
                return i, j
            num += 1
    raise ValueError(f"can not find pos")


def get_span_num(sen_len, max_span_length):
    span_label_num = 0
    for i in range(sen_len):
        span_label_num += min(max_span_length, sen_len - i)
    return span_label_num