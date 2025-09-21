#include "debug.h"
#include "color.h"

int8_t dbglvl() {
    if (_dbg_lvl == INT8_MIN) {
        errno = 0;
        const char *s = getenv("DEBUG");
        if (s != NULL) {
            char *e;
            long val;
            val = strtol(s, &e, 10);
            if (e == s || *e != '\0' || errno == EINVAL || errno == ERANGE) {
                printf("invalid value for DEBUG=\"%s\"; defaulting to 0\n", s);
                _dbg_lvl = 0;
            } else {
                _dbg_lvl = (int8_t) val;
                if (_dbg_lvl < 0) {
                    printf("invalid range for DEBUG=\"%d\"; defaulting to 0\n", _dbg_lvl);
                    _dbg_lvl = 0;
                }
            }
        } else {
            _dbg_lvl = 0;
        }
    }
    return _dbg_lvl;
}

uint8_t dbg_start(int8_t lvl, const char *name) {
    assert(_dbg_last <= UINT8_MAX);
    uint8_t id = 0;
    if (dbglvl() >= lvl) {
        id = _dbg_last++;
        if (id > 0 && _dbg_mod[id-1] == 1) printf("s\n");
        printf("%*s", id * DBG_ALIGN, ""); \
        printf(YEL "%s" RST " ", name);
        for (uint8_t i = 0; i <= id; i++) _dbg_mod[i] = 1;
    }
    return id;
}

void dbg_end(uint8_t id, int8_t lvl) {
    if (_dbg_last == 0) return;
    if (dbglvl() >= lvl) {
        if (_dbg_mod[id] == 1) printf("e\n");
        for (uint8_t i = 0; i <= id; i++) _dbg_mod[i] = 0;
        printf("%d[", id);
        for (uint8_t i = 0; i <= 5; i++) printf("%d%s", _dbg_mod[i], i < 5 ? " " : "");
        printf("]");
        _dbg_last--;
    }
}