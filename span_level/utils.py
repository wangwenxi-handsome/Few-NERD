def get_span_pos(start, end, sen_len):
    assert end >= start
    past = int(start * sen_len - start * (start - 1) / 2)
    return past + (end - start)


def get_span_start_end(pos, sen_len):
    num = 0
    for i in range(len(sen_len) - 1):
        for j in range(i, len(sen_len)):
            if num == pos:
                return i, j
            num += 1
    raise ValueError(f"can not find pos")