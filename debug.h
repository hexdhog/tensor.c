#ifndef __DEBUG_H__
#define __DEBUG_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <errno.h>
#include <assert.h>

#define DBG_ALIGN 2

static int8_t _dbg_lvl = INT8_MIN;
static uint8_t _dbg_last = 0;
static uint8_t _dbg_mod[UINT8_MAX] = { 0 };

// printf("[");
// for (uint8_t i = 0; i <= id+1; i++) printf("%d ", _dbg_mod[i]);
// printf("]");

int8_t dbglvl();
uint8_t dbg_start(int8_t lvl, const char *name);
void dbg_end(uint8_t id, int8_t lvl);
#define dbg(id, lvl, code) { \
    if (dbglvl() >= (int8_t)(lvl)) { \
        for (uint8_t i = 0; i <= id; i++) _dbg_mod[i] = 1; \
        if (_dbg_mod[id + 1] == 1) { \
            printf("\n%*s", id * DBG_ALIGN, ""); \
            _dbg_mod[id + 1] = 0; \
        } \
        do { code } while (0); \
    } \
}

// printf("%d[", id);
// for (uint8_t i = 0; i <= 5; i++) printf("%d%s", _dbg_mod[i], i < 5 ? " " : "");
// printf("]");

#endif