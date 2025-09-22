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
        // printf("s[");
        // for (uint8_t i = 0; i <= id+1; i++) printf("%d%s", _dbg_mod[i], i < id+1 ? " " : "");
        // printf("]");
        if (id > 0 && _dbg_mod[id-1] == 1) printf("\n");
        printf("%*s", id * DBG_ALIGN, ""); \
        printf(YEL "%s" RST " ", name);
        _dbg_mod[id] = 1;
        // printf("se[");
        // for (uint8_t i = 0; i <= id+1; i++) printf("%d%s", _dbg_mod[i], i < id+1 ? " " : "");
        // printf("]");
    }
    return id;
}

void dbg(uint8_t id, int8_t lvl, const char *fmt, ...) {
    if (dbglvl() >= (int8_t)(lvl)) {
        // printf("d[");
        // for (uint8_t i = 0; i <= id+1; i++) printf("%d%s", _dbg_mod[i], i < id+1 ? " " : "");
        // printf("]");
        for (uint8_t i = 0; i <= id; i++) _dbg_mod[i] = 1;
        if (_dbg_mod[id + 1] == 1) {
            printf("\n%*s", id * DBG_ALIGN, "");
            _dbg_mod[id + 1] = 0;
        }
        va_list args;
        va_start(args, fmt);
        vprintf(fmt, args);
        va_end(args);
    }
}

void dbg_end(uint8_t id, int8_t lvl) {
    if (_dbg_last == 0) return;
    if (dbglvl() >= lvl) {
        // printf("e[");
        // for (uint8_t i = 0; i <= id+1; i++) printf("%d%s", _dbg_mod[i], i < id+1 ? " " : "");
        // printf("]");
        if (_dbg_mod[id] == 1) printf("\n");
        for (uint8_t i = 0; i <= id; i++) _dbg_mod[i] = 0;
        _dbg_last--;
    }
}