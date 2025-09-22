#ifndef __DEBUG_H__
#define __DEBUG_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <errno.h>
#include <assert.h>
#include <stdarg.h>

#include "color.h"

#define DBG_ALIGN 14

static int8_t _dbg_lvl = INT8_MIN;
static volatile uint8_t _dbg_last = 0;
static volatile uint8_t _dbg_mod[UINT8_MAX] = { 0 };

#define DBG(lvl, code) { \
    if (dbglvl() >= (int8_t)(lvl)) { \
        printf(YEL "%*s " RST, DBG_ALIGN, __func__); \
        do { code } while (0); \
        printf("\n"); \
    } \
} \

int8_t dbglvl();
// uint8_t dbg_start(int8_t lvl, const char *name);
// void dbg(uint8_t id, int8_t lvl, const char *fmt, ...);
// void dbg_end(uint8_t id, int8_t lvl);

#endif