#ifndef __DEBUG_H__
#define __DEBUG_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <errno.h>
#include <assert.h>
#include <stdarg.h>

#define DBG_ALIGN 2

static int8_t _dbg_lvl = INT8_MIN;
static volatile uint8_t _dbg_last = 0;
static volatile uint8_t _dbg_mod[UINT8_MAX] = { 0 };

int8_t dbglvl();
uint8_t dbg_start(int8_t lvl, const char *name);
void dbg(uint8_t id, int8_t lvl, const char *fmt, ...);
void dbg_end(uint8_t id, int8_t lvl);

#endif