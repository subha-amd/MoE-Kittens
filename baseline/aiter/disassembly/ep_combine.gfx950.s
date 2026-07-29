
/data/subvadla-traces/kernel_evidence_e/raw/ep_intranode.gfx950.hsaco:	file format elf64-amdgpu

Disassembly of section .text:

0000000000039000 <EpCombineIntraNodeKernel_bf16_nop2p>:
	s_load_dword s33, s[0:1], 0x2e8                            // 000000039000: C0020840 000002E8
	s_add_u32 s3, s0, 0x2e8                                    // 000000039008: 8003FF00 000002E8
	s_addc_u32 s5, s1, 0                                       // 000000039010: 82058001
	v_mov_b32_e32 v1, 0                                        // 000000039014: 7E020280
	s_mov_b32 s32, 0                                           // 000000039018: BEA00080
	s_waitcnt lgkmcnt(0)                                       // 00000003901C: BF8CC07F
	s_cmp_lt_u32 s2, s33                                       // 000000039020: BF0A2102
	s_cselect_b32 s4, 12, 18                                   // 000000039024: 8504928C
	s_add_u32 s4, s3, s4                                       // 000000039028: 80040403
	s_addc_u32 s5, s5, 0                                       // 00000003902C: 82058005
	global_load_ushort v1, v1, s[4:5]                          // 000000039030: DC488000 01040001
	s_load_dwordx4 s[8:11], s[0:1], 0x288                      // 000000039038: C00A0200 00000288
	s_load_dwordx2 s[12:13], s[0:1], 0x278                     // 000000039040: C0060300 00000278
	s_load_dwordx4 s[4:7], s[0:1], 0x268                       // 000000039048: C00A0100 00000268
	s_load_dwordx4 s[52:55], s[0:1], 0x0                       // 000000039050: C00A0D00 00000000
	s_load_dword s46, s[0:1], 0x20                             // 000000039058: C0020B80 00000020
	s_load_dwordx2 s[16:17], s[0:1], 0x78                      // 000000039060: C0060400 00000078
	s_load_dwordx2 s[48:49], s[0:1], 0xa0                      // 000000039068: C0060C00 000000A0
	s_waitcnt lgkmcnt(0)                                       // 000000039070: BF8CC07F
	v_writelane_b32 v63, s4, 0                                 // 000000039074: D28A003F 00010004
	s_load_dword s3, s[6:7], 0x0                               // 00000003907C: C00200C3 00000000
	s_cmp_eq_u64 s[12:13], 0                                   // 000000039084: BF12800C
	v_writelane_b32 v63, s5, 1                                 // 000000039088: D28A003F 00010205
	v_writelane_b32 v63, s6, 2                                 // 000000039090: D28A003F 00010406
	v_writelane_b32 v63, s7, 3                                 // 000000039098: D28A003F 00010607
	s_cselect_b64 s[6:7], -1, 0                                // 0000000390A0: 858680C1
	s_cmp_lg_u64 s[12:13], 0                                   // 0000000390A4: BF13800C
	v_mov_b64_e32 v[4:5], s[12:13]                             // 0000000390A8: 7E08700C
	s_waitcnt vmcnt(0)                                         // 0000000390AC: BF8C0F70
	v_readfirstlane_b32 s4, v1                                 // 0000000390B0: 7E080501
	s_cbranch_scc1 14                                          // 0000000390B4: BF85000E <EpCombineIntraNodeKernel_bf16_nop2p+0xf0>
	s_load_dwordx2 s[12:13], s[0:1], 0x260                     // 0000000390B8: C0060300 00000260
	s_ashr_i32 s15, s52, 31                                    // 0000000390C0: 900F9F34
	s_mov_b32 s14, s52                                         // 0000000390C4: BE8E0034
	s_lshl_b64 s[14:15], s[14:15], 3                           // 0000000390C8: 8E8E830E
	s_waitcnt lgkmcnt(0)                                       // 0000000390CC: BF8CC07F
	s_load_dwordx2 s[12:13], s[12:13], 0x10                    // 0000000390D0: C0060306 00000010
	s_waitcnt lgkmcnt(0)                                       // 0000000390D8: BF8CC07F
	s_add_u32 s12, s12, s14                                    // 0000000390DC: 800C0E0C
	s_addc_u32 s13, s13, s15                                   // 0000000390E0: 820D0F0D
	v_mov_b64_e32 v[2:3], s[12:13]                             // 0000000390E4: 7E04700C
	flat_load_dwordx2 v[4:5], v[2:3]                           // 0000000390E8: DC540000 04000002
	s_load_dwordx2 s[12:13], s[10:11], 0x0                     // 0000000390F0: C0060305 00000000
	s_load_dwordx2 s[22:23], s[0:1], 0xb0                      // 0000000390F8: C0060580 000000B0
	s_load_dwordx2 s[14:15], s[0:1], 0x190                     // 000000039100: C0060380 00000190
	s_and_b32 s59, 0xffff, s4                                  // 000000039108: 863B04FF 0000FFFF
	s_lshr_b32 s58, s59, 6                                     // 000000039110: 8F3A863B
	v_lshrrev_b32_e32 v41, 6, v0                               // 000000039114: 20520086
	s_mul_i32 s4, s58, s2                                      // 000000039118: 9204023A
	s_waitcnt lgkmcnt(0)                                       // 00000003911C: BF8CC07F
	v_writelane_b32 v63, s14, 4                                // 000000039120: D28A003F 0001080E
	s_ashr_i32 s55, s54, 31                                    // 000000039128: 90379F36
	s_ashr_i32 s47, s46, 31                                    // 00000003912C: 902F9F2E
	v_writelane_b32 v63, s15, 5                                // 000000039130: D28A003F 00010A0F
	s_load_dwordx2 s[14:15], s[0:1], 0x210                     // 000000039138: C0060380 00000210
	s_load_dword s5, s[0:1], 0x18                              // 000000039140: C0020140 00000018
	v_add_u32_e32 v53, s4, v41                                 // 000000039148: 686A5204
	s_lshl_b64 s[60:61], s[54:55], 1                           // 00000003914C: 8EBC8136
	v_and_b32_e32 v42, 63, v0                                  // 000000039150: 265400BF
	s_mul_i32 s83, s58, s33                                    // 000000039154: 9253213A
	s_waitcnt lgkmcnt(0)                                       // 000000039158: BF8CC07F
	v_writelane_b32 v63, s5, 6                                 // 00000003915C: D28A003F 00010C05
	s_lshl_b64 s[4:5], s[46:47], 2                             // 000000039164: 8E84822E
	s_cmp_lg_u64 s[16:17], 0                                   // 000000039168: BF138010
	s_cselect_b64 s[62:63], -1, 0                              // 00000003916C: 85BE80C1
	s_and_b64 s[18:19], s[62:63], exec                         // 000000039170: 86927E3E
	s_cselect_b32 s4, s4, 0                                    // 000000039174: 85048004
	s_cselect_b32 s5, s5, 0                                    // 000000039178: 85058005
	s_add_u32 s56, s4, s60                                     // 00000003917C: 80383C04
	s_addc_u32 s57, s5, s61                                    // 000000039180: 82393D05
	v_cmp_gt_i32_e32 vcc, s3, v53                              // 000000039184: 7D886A03
	v_lshlrev_b32_e32 v2, 3, v0                                // 000000039188: 24040083
	s_and_saveexec_b64 s[18:19], vcc                           // 00000003918C: BE92206A
	s_cbranch_execz 254                                        // 000000039190: BF8800FE <EpCombineIntraNodeKernel_bf16_nop2p+0x58c>
	s_load_dwordx2 s[4:5], s[0:1], 0x68                        // 000000039194: C0060100 00000068
	s_load_dword s28, s[0:1], 0x18                             // 00000003919C: C0020700 00000018
	s_cmpk_gt_u32 s54, 0x1ff                                   // 0000000391A4: B53601FF
	s_cselect_b64 s[36:37], -1, 0                              // 0000000391A8: 85A480C1
	s_lshr_b64 s[24:25], s[54:55], 9                           // 0000000391AC: 8F988936
	s_cmpk_gt_u32 s46, 0xff                                    // 0000000391B0: B52E00FF
	s_waitcnt lgkmcnt(0)                                       // 0000000391B4: BF8CC07F
	s_mul_i32 s64, s28, s53                                    // 0000000391B8: 9240351C
	s_cselect_b64 s[26:27], -1, 0                              // 0000000391BC: 859A80C1
	s_abs_i32 s65, s64                                         // 0000000391C0: BEC13040
	v_cvt_f32_u32_e32 v1, s65                                  // 0000000391C4: 7E020C41
	v_lshlrev_b32_e32 v3, 2, v0                                // 0000000391C8: 24060082
	s_sub_i32 s30, 0, s65                                      // 0000000391CC: 819E4180
	s_load_dwordx2 s[20:21], s[48:49], 0x10                    // 0000000391D0: C0060518 00000010
	v_rcp_iflag_f32_e32 v1, v1                                 // 0000000391D8: 7E024701
	v_and_b32_e32 v18, 0xfc, v3                                // 0000000391DC: 262406FF 000000FC
	v_mov_b32_e32 v7, 0                                        // 0000000391E4: 7E0E0280
	v_lshlrev_b32_e32 v6, 4, v42                               // 0000000391E8: 240C5484
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 0000000391EC: 0A0202FF 4F7FFFFE
	v_cvt_u32_f32_e32 v1, v1                                   // 0000000391F4: 7E020F01
	v_and_b32_e32 v16, 0x1f8, v2                               // 0000000391F8: 262004FF 000001F8
	v_lshl_add_u64 v[12:13], s[16:17], 0, v[6:7]               // 000000039200: D208000C 04190010
	s_mul_i32 s68, s33, s46                                    // 000000039208: 92442E21
	v_mul_lo_u32 v3, s30, v1                                   // 00000003920C: D2850003 0002021E
	v_mad_i64_i32 v[8:9], s[30:31], s54, v53, 0                // 000000039214: D1E91E08 02026A36
	v_lshlrev_b64 v[10:11], 1, v[8:9]                          // 00000003921C: D28F000A 00021081
	v_mul_hi_u32 v3, v1, v3                                    // 000000039224: D2860003 00020701
	v_lshl_add_u64 v[8:9], v[10:11], 0, v[6:7]                 // 00000003922C: D2080008 0419010A
	v_add_u32_e32 v1, v1, v3                                   // 000000039234: 68020701
	v_lshl_add_u64 v[8:9], s[4:5], 0, v[8:9]                   // 000000039238: D2080008 04210004
	s_mul_hi_i32 s31, s54, s83                                 // 000000039240: 969F5336
	s_mul_i32 s30, s54, s83                                    // 000000039244: 921E5336
	v_cndmask_b32_e64 v3, 0, 1, s[36:37]                       // 000000039248: D1000003 00910280
	s_mul_i32 s66, s28, s52                                    // 000000039250: 9242341C
	v_mov_b32_e32 v43, v7                                      // 000000039254: 7E560307
	s_lshr_b64 s[28:29], s[46:47], 8                           // 000000039258: 8F9C882E
	s_ashr_i32 s67, s64, 31                                    // 00000003925C: 90439F40
	v_lshl_add_u64 v[8:9], v[8:9], 0, 8                        // 000000039260: D2080008 02210108
	s_lshl_b64 s[30:31], s[30:31], 1                           // 000000039268: 8E9E811E
	v_lshl_add_u64 v[10:11], s[4:5], 0, v[10:11]               // 00000003926C: D208000A 04290004
	v_lshl_add_u64 v[12:13], v[12:13], 0, 8                    // 000000039274: D208000C 0221010C
	v_mul_lo_u32 v14, s46, v53                                 // 00000003927C: D285000E 00026A2E
	s_mul_i32 s68, s68, s58                                    // 000000039284: 92443A44
	s_mov_b64 s[34:35], 0                                      // 000000039288: BEA20180
	v_cmp_ne_u32_e64 s[4:5], 1, v3                             // 00000003928C: D0CD0004 00020681
	v_lshlrev_b32_e32 v6, 1, v16                               // 000000039294: 240C2081
	s_mov_b64 s[36:37], 0x400                                  // 000000039298: BEA401FF 00000400
	s_mov_b64 s[38:39], 0x80                                   // 0000000392A0: BEA601FF 00000080
	v_lshlrev_b32_e32 v16, 2, v18                              // 0000000392A8: 24202482
	s_mov_b64 s[40:41], 0x100                                  // 0000000392AC: BEA801FF 00000100
	v_mov_b32_e32 v18, v53                                     // 0000000392B4: 7E240335
	s_branch 11                                                // 0000000392B8: BF82000B <EpCombineIntraNodeKernel_bf16_nop2p+0x2e8>
	s_or_b64 exec, exec, s[42:43]                              // 0000000392BC: 87FE2A7E
	v_add_u32_e32 v18, s83, v18                                // 0000000392C0: 68242453
	v_cmp_le_i32_e32 vcc, s3, v18                              // 0000000392C4: 7D862403
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[30:31]                 // 0000000392C8: D2080008 00790108
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[30:31]             // 0000000392D0: D208000A 0079010A
	s_or_b64 s[34:35], vcc, s[34:35]                           // 0000000392D8: 87A2226A
	v_add_u32_e32 v14, s68, v14                                // 0000000392DC: 681C1C44
	s_andn2_b64 exec, exec, s[34:35]                           // 0000000392E0: 89FE227E
	s_cbranch_execz 169                                        // 0000000392E4: BF8800A9 <EpCombineIntraNodeKernel_bf16_nop2p+0x58c>
	v_ashrrev_i32_e32 v19, 31, v18                             // 0000000392E8: 2226249F
	s_waitcnt vmcnt(0)                                         // 0000000392EC: BF8C0F70
	v_lshl_add_u64 v[20:21], v[18:19], 2, v[4:5]               // 0000000392F0: D2080014 04110512
	flat_load_dword v3, v[20:21]                               // 0000000392F8: DC500000 03000014
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 000000039300: BF8C0070
	v_sub_u32_e32 v17, 0, v3                                   // 000000039304: 6A220680
	v_max_i32_e32 v17, v3, v17                                 // 000000039308: 1A222303
	v_mul_hi_u32 v19, v17, v1                                  // 00000003930C: D2860013 00020311
	v_mul_lo_u32 v20, v19, s65                                 // 000000039314: D2850014 00008313
	v_sub_u32_e32 v17, v17, v20                                // 00000003931C: 6A222911
	v_add_u32_e32 v21, 1, v19                                  // 000000039320: 682A2681
	v_cmp_le_u32_e32 vcc, s65, v17                             // 000000039324: 7D962241
	v_subrev_u32_e32 v20, s65, v17                             // 000000039328: 6C282241
	v_ashrrev_i32_e32 v15, 31, v3                              // 00000003932C: 221E069F
	v_cndmask_b32_e32 v19, v19, v21, vcc                       // 000000039330: 00262B13
	v_cndmask_b32_e32 v17, v17, v20, vcc                       // 000000039334: 00222911
	v_add_u32_e32 v20, 1, v19                                  // 000000039338: 68282681
	v_cmp_le_u32_e32 vcc, s65, v17                             // 00000003933C: 7D962241
	v_xor_b32_e32 v15, s67, v15                                // 000000039340: 2A1E1E43
	s_nop 0                                                    // 000000039344: BF800000
	v_cndmask_b32_e32 v17, v19, v20, vcc                       // 000000039348: 00222913
	v_xor_b32_e32 v17, v17, v15                                // 00000003934C: 2A221F11
	v_sub_u32_e32 v22, v17, v15                                // 000000039350: 6A2C1F11
	v_ashrrev_i32_e32 v23, 31, v22                             // 000000039354: 222E2C9F
	v_lshl_add_u64 v[20:21], v[22:23], 3, s[20:21]             // 000000039358: D2080014 00510716
	flat_load_dwordx2 v[20:21], v[20:21]                       // 000000039360: DC540000 14000014
	v_mul_lo_u32 v15, v22, s64                                 // 000000039368: D285000F 00008116
	v_sub_u32_e32 v3, v3, v15                                  // 000000039370: 6A061F03
	v_add_u32_e32 v3, s66, v3                                  // 000000039374: 68060642
	v_ashrrev_i32_e32 v17, 31, v3                              // 000000039378: 2222069F
	v_mul_lo_u32 v15, s57, v3                                  // 00000003937C: D285000F 00020639
	v_mul_lo_u32 v17, s56, v17                                 // 000000039384: D2850011 00022238
	s_and_b64 vcc, exec, s[4:5]                                // 00000003938C: 86EA047E
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 000000039390: BF8C0070
	v_mad_u64_u32 v[24:25], s[42:43], s56, v3, v[20:21]        // 000000039394: D1E82A18 04520638
	v_add3_u32 v25, v15, v25, v17                              // 00000003939C: D1FF0019 0446330F
	s_mov_b64 s[42:43], 0                                      // 0000000393A4: BEAA0180
	s_cbranch_vccnz 20                                         // 0000000393A8: BF870014 <EpCombineIntraNodeKernel_bf16_nop2p+0x3fc>
	v_lshl_add_u64 v[22:23], v[24:25], 0, v[6:7]               // 0000000393AC: D2080016 04190118
	v_mov_b64_e32 v[26:27], v[8:9]                             // 0000000393B4: 7E347108
	s_mov_b64 s[44:45], s[24:25]                               // 0000000393B8: BEAC0118
	global_load_dwordx4 v[28:31], v[26:27], off offset:-8 nt   // 0000000393BC: DC5E9FF8 1C7F001A
	s_add_u32 s42, s42, 0x200                                  // 0000000393C4: 802AFF2A 00000200
	s_addc_u32 s43, s43, 0                                     // 0000000393CC: 822B802B
	s_add_u32 s44, s44, -1                                     // 0000000393D0: 802CC12C
	s_addc_u32 s45, s45, -1                                    // 0000000393D4: 822DC12D
	v_lshl_add_u64 v[26:27], v[26:27], 0, s[36:37]             // 0000000393D8: D208001A 0091011A
	s_cmp_lg_u64 s[44:45], 0                                   // 0000000393E0: BF13802C
	s_waitcnt vmcnt(0)                                         // 0000000393E4: BF8C0F70
	flat_store_dwordx4 v[22:23], v[28:31]                      // 0000000393E8: DC7C0000 00001C16
	v_lshl_add_u64 v[22:23], v[22:23], 0, s[36:37]             // 0000000393F0: D2080016 00910116
	s_cbranch_scc1 65520                                       // 0000000393F8: BF85FFF0 <EpCombineIntraNodeKernel_bf16_nop2p+0x3bc>
	v_mad_u64_u32 v[22:23], s[44:45], s56, v3, 0               // 0000000393FC: D1E82C16 02020638
	v_lshl_add_u64 v[26:27], s[42:43], 0, v[42:43]             // 000000039404: D208001A 04A9002A
	v_add3_u32 v23, v23, v17, v15                              // 00000003940C: D1FF0017 043E2317
	v_cmp_gt_u64_e32 vcc, s[54:55], v[26:27]                   // 000000039414: 7DD83436
	s_and_saveexec_b64 s[42:43], vcc                           // 000000039418: BEAA206A
	s_cbranch_execz 24                                         // 00000003941C: BF880018 <EpCombineIntraNodeKernel_bf16_nop2p+0x480>
	v_lshlrev_b64 v[30:31], 1, v[26:27]                        // 000000039420: D28F001E 00023481
	v_lshl_add_u64 v[28:29], v[10:11], 0, v[30:31]             // 000000039428: D208001C 0479010A
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[30:31]             // 000000039430: D208001E 04790116
	v_lshl_add_u64 v[30:31], v[20:21], 0, v[30:31]             // 000000039438: D208001E 04790114
	s_mov_b64 s[44:45], 0                                      // 000000039440: BEAC0180
	global_load_ushort v3, v[28:29], off                       // 000000039444: DC488000 037F001C
	v_lshl_add_u64 v[26:27], v[26:27], 0, 64                   // 00000003944C: D208001A 0301011A
	v_cmp_le_u64_e32 vcc, s[54:55], v[26:27]                   // 000000039454: 7DD63436
	v_lshl_add_u64 v[28:29], v[28:29], 0, s[38:39]             // 000000039458: D208001C 0099011C
	s_or_b64 s[44:45], vcc, s[44:45]                           // 000000039460: 87AC2C6A
	s_waitcnt vmcnt(0)                                         // 000000039464: BF8C0F70
	flat_store_short v[30:31], v3                              // 000000039468: DC680000 0000031E
	v_lshl_add_u64 v[30:31], v[30:31], 0, s[38:39]             // 000000039470: D208001E 0099011E
	s_andn2_b64 exec, exec, s[44:45]                           // 000000039478: 89FE2C7E
	s_cbranch_execnz 65521                                     // 00000003947C: BF89FFF1 <EpCombineIntraNodeKernel_bf16_nop2p+0x444>
	s_or_b64 exec, exec, s[42:43]                              // 000000039480: 87FE2A7E
	s_andn2_b64 vcc, exec, s[62:63]                            // 000000039484: 89EA3E7E
	s_cbranch_vccnz 65421                                      // 000000039488: BF87FF8D <EpCombineIntraNodeKernel_bf16_nop2p+0x2c0>
	v_ashrrev_i32_e32 v15, 31, v14                             // 00000003948C: 221E1C9F
	v_lshlrev_b64 v[26:27], 2, v[14:15]                        // 000000039490: D28F001A 00021C82
	s_andn2_b64 vcc, exec, s[26:27]                            // 000000039498: 89EA1A7E
	s_mov_b64 s[42:43], 0                                      // 00000003949C: BEAA0180
	s_cbranch_vccnz 24                                         // 0000000394A0: BF870018 <EpCombineIntraNodeKernel_bf16_nop2p+0x504>
	v_lshl_add_u64 v[28:29], v[24:25], 0, s[60:61]             // 0000000394A4: D208001C 00F10118
	v_mov_b32_e32 v17, v7                                      // 0000000394AC: 7E220307
	v_lshl_add_u64 v[24:25], v[12:13], 0, v[26:27]             // 0000000394B0: D2080018 0469010C
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[16:17]             // 0000000394B8: D208001C 0441011C
	s_mov_b64 s[44:45], s[28:29]                               // 0000000394C0: BEAC011C
	global_load_dwordx4 v[30:33], v[24:25], off offset:-8 nt   // 0000000394C4: DC5E9FF8 1E7F0018
	s_add_u32 s42, s42, 0x100                                  // 0000000394CC: 802AFF2A 00000100
	s_addc_u32 s43, s43, 0                                     // 0000000394D4: 822B802B
	s_add_u32 s44, s44, -1                                     // 0000000394D8: 802CC12C
	s_addc_u32 s45, s45, -1                                    // 0000000394DC: 822DC12D
	v_lshl_add_u64 v[24:25], v[24:25], 0, s[36:37]             // 0000000394E0: D2080018 00910118
	s_cmp_lg_u64 s[44:45], 0                                   // 0000000394E8: BF13802C
	s_waitcnt vmcnt(0)                                         // 0000000394EC: BF8C0F70
	flat_store_dwordx4 v[28:29], v[30:33]                      // 0000000394F0: DC7C0000 00001E1C
	v_lshl_add_u64 v[28:29], v[28:29], 0, s[36:37]             // 0000000394F8: D208001C 0091011C
	s_cbranch_scc1 65520                                       // 000000039500: BF85FFF0 <EpCombineIntraNodeKernel_bf16_nop2p+0x4c4>
	v_lshl_add_u64 v[24:25], s[42:43], 0, v[42:43]             // 000000039504: D2080018 04A9002A
	v_cmp_gt_u64_e32 vcc, s[46:47], v[24:25]                   // 00000003950C: 7DD8302E
	s_and_saveexec_b64 s[42:43], vcc                           // 000000039510: BEAA206A
	s_cbranch_execz 65385                                      // 000000039514: BF88FF69 <EpCombineIntraNodeKernel_bf16_nop2p+0x2bc>
	v_lshlrev_b64 v[28:29], 2, v[24:25]                        // 000000039518: D28F001C 00023082
	v_lshl_add_u64 v[26:27], v[28:29], 0, v[26:27]             // 000000039520: D208001A 0469011C
	v_lshl_add_u64 v[20:21], v[20:21], 0, s[60:61]             // 000000039528: D2080014 00F10114
	v_lshl_add_u64 v[22:23], v[22:23], 0, v[28:29]             // 000000039530: D2080016 04710116
	v_lshl_add_u64 v[26:27], s[16:17], 0, v[26:27]             // 000000039538: D208001A 04690010
	v_lshl_add_u64 v[20:21], v[20:21], 0, v[22:23]             // 000000039540: D2080014 04590114
	s_mov_b64 s[44:45], 0                                      // 000000039548: BEAC0180
	global_load_dword v3, v[26:27], off                        // 00000003954C: DC508000 037F001A
	v_lshl_add_u64 v[24:25], v[24:25], 0, 64                   // 000000039554: D2080018 03010118
	v_cmp_le_u64_e32 vcc, s[46:47], v[24:25]                   // 00000003955C: 7DD6302E
	v_lshl_add_u64 v[26:27], v[26:27], 0, s[40:41]             // 000000039560: D208001A 00A1011A
	s_or_b64 s[44:45], vcc, s[44:45]                           // 000000039568: 87AC2C6A
	s_waitcnt vmcnt(0)                                         // 00000003956C: BF8C0F70
	flat_store_dword v[20:21], v3                              // 000000039570: DC700000 00000314
	v_lshl_add_u64 v[20:21], v[20:21], 0, s[40:41]             // 000000039578: D2080014 00A10114
	s_andn2_b64 exec, exec, s[44:45]                           // 000000039580: 89FE2C7E
	s_cbranch_execnz 65521                                     // 000000039584: BF89FFF1 <EpCombineIntraNodeKernel_bf16_nop2p+0x54c>
	s_branch 65356                                             // 000000039588: BF82FF4C <EpCombineIntraNodeKernel_bf16_nop2p+0x2bc>
	s_or_b64 exec, exec, s[18:19]                              // 00000003958C: 87FE127E
	s_load_dword s21, s[0:1], 0x58                             // 000000039590: C0020540 00000058
	v_cmp_eq_u32_e32 vcc, 0, v0                                // 000000039598: 7D940080
	s_waitcnt lgkmcnt(0)                                       // 00000003959C: BF8CC07F
	s_barrier                                                  // 0000000395A0: BF8A0000
	s_and_saveexec_b64 s[4:5], vcc                             // 0000000395A4: BE84206A
	s_cbranch_execz 14                                         // 0000000395A8: BF88000E <EpCombineIntraNodeKernel_bf16_nop2p+0x5e4>
	s_mov_b64 s[16:17], exec                                   // 0000000395AC: BE90017E
	v_mbcnt_lo_u32_b32 v1, s16, 0                              // 0000000395B0: D28C0001 00010010
	v_mbcnt_hi_u32_b32 v1, s17, v1                             // 0000000395B8: D28D0001 00020211
	v_cmp_eq_u32_e32 vcc, 0, v1                                // 0000000395C0: 7D940280
	s_and_b64 s[18:19], exec, vcc                              // 0000000395C4: 86926A7E
	s_mov_b64 exec, s[18:19]                                   // 0000000395C8: BEFE0112
	s_cbranch_execz 5                                          // 0000000395CC: BF880005 <EpCombineIntraNodeKernel_bf16_nop2p+0x5e4>
	s_bcnt1_i32_b64 s3, s[16:17]                               // 0000000395D0: BE830D10
	v_mov_b32_e32 v1, 0                                        // 0000000395D4: 7E020280
	v_mov_b32_e32 v3, s3                                       // 0000000395D8: 7E060203
	global_atomic_add v1, v3, s[14:15]                         // 0000000395DC: DD088000 000E0301
	s_or_b64 exec, exec, s[4:5]                                // 0000000395E4: 87FE047E
	s_mul_i32 s2, s2, s59                                      // 0000000395E8: 92023B02
	s_waitcnt vmcnt(0)                                         // 0000000395EC: BF8C0F70
	v_add_u32_e32 v4, s2, v0                                   // 0000000395F0: 68080002
	v_cmp_gt_i32_e32 vcc, s53, v4                              // 0000000395F4: 7D880835
	s_and_saveexec_b64 s[2:3], vcc                             // 0000000395F8: BE82206A
	s_cbranch_execz 30                                         // 0000000395FC: BF88001E <EpCombineIntraNodeKernel_bf16_nop2p+0x678>
	v_mov_b32_e32 v1, 0                                        // 000000039600: 7E020280
	global_load_dword v3, v1, s[14:15] sc0 sc1                 // 000000039604: DE518000 030E0001
	s_waitcnt vmcnt(0)                                         // 00000003960C: BF8C0F70
	v_cmp_ne_u32_e32 vcc, s33, v3                              // 000000039610: 7D9A0621
	s_cbranch_vccnz 65531                                      // 000000039614: BF87FFFB <EpCombineIntraNodeKernel_bf16_nop2p+0x604>
	v_mov_b32_e32 v1, 0                                        // 000000039618: 7E020280
	global_store_dword v1, v1, s[14:15] sc1                    // 00000003961C: DE708000 000E0101
	buffer_wbl2 sc0 sc1                                        // 000000039624: E0A0C000 00000000
	s_waitcnt vmcnt(0)                                         // 00000003962C: BF8C0F70
	buffer_inv sc0 sc1                                         // 000000039630: E0A4C000 00000000
	global_load_dwordx2 v[6:7], v1, s[8:9] offset:16           // 000000039638: DC548010 06080001
	v_ashrrev_i32_e32 v5, 31, v4                               // 000000039640: 220A089F
	s_ashr_i32 s5, s52, 31                                     // 000000039644: 90059F34
	s_mov_b32 s4, s52                                          // 000000039648: BE840034
	v_mov_b64_e32 v[8:9], s[12:13]                             // 00000003964C: 7E10700C
	s_waitcnt vmcnt(0)                                         // 000000039650: BF8C0F70
	v_lshl_add_u64 v[6:7], v[4:5], 3, v[6:7]                   // 000000039654: D2080006 04190704
	flat_load_dwordx2 v[6:7], v[6:7]                           // 00000003965C: DC540000 06000006
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 000000039664: BF8C0070
	v_lshl_add_u64 v[6:7], s[4:5], 3, v[6:7]                   // 000000039668: D2080006 04190604
	flat_store_dwordx2 v[6:7], v[8:9] sc0 sc1                  // 000000039670: DE750000 00000806
	s_or_b64 exec, exec, s[2:3]                                // 000000039678: 87FE027E
	v_cmp_eq_u32_e32 vcc, 0, v4                                // 00000003967C: 7D940880
	s_and_saveexec_b64 s[2:3], vcc                             // 000000039680: BE82206A
	s_cbranch_execz 14                                         // 000000039684: BF88000E <EpCombineIntraNodeKernel_bf16_nop2p+0x6c0>
	s_mov_b64 s[4:5], exec                                     // 000000039688: BE84017E
	v_mbcnt_lo_u32_b32 v1, s4, 0                               // 00000003968C: D28C0001 00010004
	v_mbcnt_hi_u32_b32 v1, s5, v1                              // 000000039694: D28D0001 00020205
	v_cmp_eq_u32_e32 vcc, 0, v1                                // 00000003969C: 7D940280
	s_and_b64 s[14:15], exec, vcc                              // 0000000396A0: 868E6A7E
	s_mov_b64 exec, s[14:15]                                   // 0000000396A4: BEFE010E
	s_cbranch_execz 5                                          // 0000000396A8: BF880005 <EpCombineIntraNodeKernel_bf16_nop2p+0x6c0>
	s_bcnt1_i32_b64 s4, s[4:5]                                 // 0000000396AC: BE840D04
	v_mov_b32_e32 v4, s4                                       // 0000000396B0: 7E080204
	v_mov_b32_e32 v5, 0                                        // 0000000396B4: 7E0A0280
	global_atomic_add_x2 v5, v[4:5], s[10:11]                  // 0000000396B8: DD888000 000A0405
	s_or_b64 exec, exec, s[2:3]                                // 0000000396C0: 87FE027E
	v_cmp_gt_i32_e32 vcc, s53, v0                              // 0000000396C4: 7D880035
	s_and_saveexec_b64 s[2:3], vcc                             // 0000000396C8: BE82206A
	s_cbranch_execz 14                                         // 0000000396CC: BF88000E <EpCombineIntraNodeKernel_bf16_nop2p+0x708>
	v_mov_b32_e32 v3, 0                                        // 0000000396D0: 7E060280
	global_load_dwordx2 v[4:5], v3, s[8:9]                     // 0000000396D4: DC548000 04080003
	s_mov_b64 s[4:5], 0                                        // 0000000396DC: BE840180
	s_waitcnt vmcnt(0)                                         // 0000000396E0: BF8C0F70
	v_lshl_add_u64 v[2:3], v[4:5], 0, v[2:3]                   // 0000000396E4: D2080002 04090104
	flat_load_dwordx2 v[4:5], v[2:3] sc0 sc1                   // 0000000396EC: DE550000 04000002
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 0000000396F4: BF8C0070
	v_cmp_eq_u64_e32 vcc, s[12:13], v[4:5]                     // 0000000396F8: 7DD4080C
	s_or_b64 s[4:5], vcc, s[4:5]                               // 0000000396FC: 8784046A
	s_andn2_b64 exec, exec, s[4:5]                             // 000000039700: 89FE047E
	s_cbranch_execnz 65529                                     // 000000039704: BF89FFF9 <EpCombineIntraNodeKernel_bf16_nop2p+0x6ec>
	s_or_b64 exec, exec, s[2:3]                                // 000000039708: 87FE027E
	s_and_b64 vcc, exec, s[6:7]                                // 00000003970C: 86EA067E
	s_waitcnt lgkmcnt(0)                                       // 000000039710: BF8CC07F
	s_barrier                                                  // 000000039714: BF8A0000
	s_cbranch_vccz 6                                           // 000000039718: BF860006 <EpCombineIntraNodeKernel_bf16_nop2p+0x734>
	s_load_dwordx4 s[4:7], s[0:1], 0x268                       // 00000003971C: C00A0100 00000268
	v_mov_b32_e32 v1, 0                                        // 000000039724: 7E020280
	s_waitcnt lgkmcnt(0)                                       // 000000039728: BF8CC07F
	global_store_dword v1, v1, s[6:7]                          // 00000003972C: DC708000 00060101
	s_cmp_eq_u32 s21, 0                                        // 000000039734: BF068015
	s_mov_b32 s4, 0                                            // 000000039738: BE840080
	s_cbranch_scc1 3653                                        // 00000003973C: BF850E45 <EpCombineIntraNodeKernel_bf16_nop2p+0x4054>
	s_abs_i32 s2, s21                                          // 000000039740: BE823015
	v_cvt_f32_u32_e32 v1, s2                                   // 000000039744: 7E020C02
	s_add_i32 s3, s21, s83                                     // 000000039748: 81035315
	s_add_i32 s5, s3, -1                                       // 00000003974C: 8105C103
	s_sub_i32 s3, 1, s3                                        // 000000039750: 81830381
	v_rcp_iflag_f32_e32 v1, v1                                 // 000000039754: 7E024701
	s_xor_b32 s7, s5, s21                                      // 000000039758: 88071505
	s_sub_i32 s6, 0, s2                                        // 00000003975C: 81860280
	s_max_i32 s3, s5, s3                                       // 000000039760: 84030305
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000039764: 0A0202FF 4F7FFFFE
	v_cvt_u32_f32_e32 v1, v1                                   // 00000003976C: 7E020F01
	s_ashr_i32 s5, s7, 31                                      // 000000039770: 90059F07
	v_readfirstlane_b32 s7, v1                                 // 000000039774: 7E0E0501
	s_mul_i32 s6, s6, s7                                       // 000000039778: 92060706
	s_mul_hi_u32 s6, s7, s6                                    // 00000003977C: 96060607
	s_add_i32 s7, s7, s6                                       // 000000039780: 81070607
	s_mul_hi_u32 s6, s3, s7                                    // 000000039784: 96060703
	s_mul_i32 s7, s6, s2                                       // 000000039788: 92070206
	s_sub_i32 s3, s3, s7                                       // 00000003978C: 81830703
	s_add_i32 s8, s6, 1                                        // 000000039790: 81088106
	s_sub_i32 s7, s3, s2                                       // 000000039794: 81870203
	s_cmp_ge_u32 s3, s2                                        // 000000039798: BF090203
	s_cselect_b32 s6, s8, s6                                   // 00000003979C: 85060608
	s_cselect_b32 s3, s7, s3                                   // 0000000397A0: 85030307
	s_add_i32 s7, s6, 1                                        // 0000000397A4: 81078106
	s_cmp_ge_u32 s3, s2                                        // 0000000397A8: BF090203
	s_cselect_b32 s2, s7, s6                                   // 0000000397AC: 85020607
	s_xor_b32 s2, s2, s5                                       // 0000000397B0: 88020502
	s_sub_i32 s50, s2, s5                                      // 0000000397B4: 81B20502
	s_ashr_i32 s51, s50, 31                                    // 0000000397B8: 90339F32
	s_add_u32 s2, s54, s50                                     // 0000000397BC: 80023236
	s_addc_u32 s3, s55, s51                                    // 0000000397C0: 82033337
	s_add_u32 s2, s2, -1                                       // 0000000397C4: 8002C102
	s_addc_u32 s3, s3, -1                                      // 0000000397C8: 8203C103
	s_or_b64 s[6:7], s[2:3], s[50:51]                          // 0000000397CC: 87863202
	s_mov_b32 s5, s7                                           // 0000000397D0: BE850007
	s_cmp_lg_u64 s[4:5], 0                                     // 0000000397D4: BF138004
	s_mov_b64 s[4:5], -1                                       // 0000000397D8: BE8401C1
	s_cbranch_scc0 3614                                        // 0000000397DC: BF840E1E <EpCombineIntraNodeKernel_bf16_nop2p+0x4058>
	v_cvt_f32_u32_e32 v1, s50                                  // 0000000397E0: 7E020C32
	v_cvt_f32_u32_e32 v2, s51                                  // 0000000397E4: 7E040C33
	s_sub_u32 s8, 0, s50                                       // 0000000397E8: 80883280
	s_subb_u32 s9, 0, s51                                      // 0000000397EC: 82893380
	v_fmamk_f32 v1, v2, 0x4f800000, v1                         // 0000000397F0: 2E020302 4F800000
	v_rcp_f32_e32 v1, v1                                       // 0000000397F8: 7E024501
	s_nop 0                                                    // 0000000397FC: BF800000
	v_mul_f32_e32 v1, 0x5f7ffffc, v1                           // 000000039800: 0A0202FF 5F7FFFFC
	v_mul_f32_e32 v2, 0x2f800000, v1                           // 000000039808: 0A0402FF 2F800000
	v_trunc_f32_e32 v2, v2                                     // 000000039810: 7E043902
	v_fmamk_f32 v1, v2, 0xcf800000, v1                         // 000000039814: 2E020302 CF800000
	v_cvt_u32_f32_e32 v2, v2                                   // 00000003981C: 7E040F02
	v_cvt_u32_f32_e32 v1, v1                                   // 000000039820: 7E020F01
	v_readfirstlane_b32 s10, v2                                // 000000039824: 7E140502
	v_readfirstlane_b32 s6, v1                                 // 000000039828: 7E0C0501
	s_mul_i32 s7, s8, s10                                      // 00000003982C: 92070A08
	s_mul_hi_u32 s12, s8, s6                                   // 000000039830: 960C0608
	s_mul_i32 s11, s9, s6                                      // 000000039834: 920B0609
	s_add_i32 s7, s12, s7                                      // 000000039838: 8107070C
	s_add_i32 s7, s7, s11                                      // 00000003983C: 81070B07
	s_mul_i32 s13, s8, s6                                      // 000000039840: 920D0608
	s_mul_i32 s12, s6, s7                                      // 000000039844: 920C0706
	s_mul_hi_u32 s14, s6, s13                                  // 000000039848: 960E0D06
	s_mul_hi_u32 s11, s6, s7                                   // 00000003984C: 960B0706
	s_add_u32 s12, s14, s12                                    // 000000039850: 800C0C0E
	s_addc_u32 s11, 0, s11                                     // 000000039854: 820B0B80
	s_mul_hi_u32 s15, s10, s13                                 // 000000039858: 960F0D0A
	s_mul_i32 s13, s10, s13                                    // 00000003985C: 920D0D0A
	s_add_u32 s12, s12, s13                                    // 000000039860: 800C0D0C
	s_mul_hi_u32 s14, s10, s7                                  // 000000039864: 960E070A
	s_addc_u32 s11, s11, s15                                   // 000000039868: 820B0F0B
	s_addc_u32 s12, s14, 0                                     // 00000003986C: 820C800E
	s_mul_i32 s7, s10, s7                                      // 000000039870: 9207070A
	s_add_u32 s7, s11, s7                                      // 000000039874: 8007070B
	s_addc_u32 s11, 0, s12                                     // 000000039878: 820B0C80
	s_add_u32 s12, s6, s7                                      // 00000003987C: 800C0706
	s_cselect_b64 s[6:7], -1, 0                                // 000000039880: 858680C1
	s_cmp_lg_u64 s[6:7], 0                                     // 000000039884: BF138006
	s_addc_u32 s10, s10, s11                                   // 000000039888: 820A0B0A
	s_mul_i32 s6, s8, s10                                      // 00000003988C: 92060A08
	s_mul_hi_u32 s7, s8, s12                                   // 000000039890: 96070C08
	s_add_i32 s6, s7, s6                                       // 000000039894: 81060607
	s_mul_i32 s9, s9, s12                                      // 000000039898: 92090C09
	s_add_i32 s6, s6, s9                                       // 00000003989C: 81060906
	s_mul_i32 s8, s8, s12                                      // 0000000398A0: 92080C08
	s_mul_hi_u32 s9, s10, s8                                   // 0000000398A4: 9609080A
	s_mul_i32 s11, s10, s8                                     // 0000000398A8: 920B080A
	s_mul_i32 s14, s12, s6                                     // 0000000398AC: 920E060C
	s_mul_hi_u32 s8, s12, s8                                   // 0000000398B0: 9608080C
	s_mul_hi_u32 s13, s12, s6                                  // 0000000398B4: 960D060C
	s_add_u32 s8, s8, s14                                      // 0000000398B8: 80080E08
	s_addc_u32 s13, 0, s13                                     // 0000000398BC: 820D0D80
	s_add_u32 s8, s8, s11                                      // 0000000398C0: 80080B08
	s_mul_hi_u32 s7, s10, s6                                   // 0000000398C4: 9607060A
	s_addc_u32 s8, s13, s9                                     // 0000000398C8: 8208090D
	s_addc_u32 s7, s7, 0                                       // 0000000398CC: 82078007
	s_mul_i32 s6, s10, s6                                      // 0000000398D0: 9206060A
	s_add_u32 s6, s8, s6                                       // 0000000398D4: 80060608
	s_addc_u32 s8, 0, s7                                       // 0000000398D8: 82080780
	s_add_u32 s9, s12, s6                                      // 0000000398DC: 8009060C
	s_cselect_b64 s[6:7], -1, 0                                // 0000000398E0: 858680C1
	s_cmp_lg_u64 s[6:7], 0                                     // 0000000398E4: BF138006
	s_addc_u32 s6, s10, s8                                     // 0000000398E8: 8206080A
	s_mul_i32 s8, s2, s6                                       // 0000000398EC: 92080602
	s_mul_hi_u32 s10, s2, s9                                   // 0000000398F0: 960A0902
	s_mul_hi_u32 s7, s2, s6                                    // 0000000398F4: 96070602
	s_add_u32 s8, s10, s8                                      // 0000000398F8: 8008080A
	s_addc_u32 s7, 0, s7                                       // 0000000398FC: 82070780
	s_mul_hi_u32 s11, s3, s9                                   // 000000039900: 960B0903
	s_mul_i32 s9, s3, s9                                       // 000000039904: 92090903
	s_add_u32 s8, s8, s9                                       // 000000039908: 80080908
	s_mul_hi_u32 s10, s3, s6                                   // 00000003990C: 960A0603
	s_addc_u32 s7, s7, s11                                     // 000000039910: 82070B07
	s_addc_u32 s8, s10, 0                                      // 000000039914: 8208800A
	s_mul_i32 s6, s3, s6                                       // 000000039918: 92060603
	s_add_u32 s10, s7, s6                                      // 00000003991C: 800A0607
	s_addc_u32 s11, 0, s8                                      // 000000039920: 820B0880
	s_mul_i32 s6, s50, s11                                     // 000000039924: 92060B32
	s_mul_hi_u32 s7, s50, s10                                  // 000000039928: 96070A32
	s_add_i32 s6, s7, s6                                       // 00000003992C: 81060607
	s_mul_i32 s7, s51, s10                                     // 000000039930: 92070A33
	s_add_i32 s12, s6, s7                                      // 000000039934: 810C0706
	s_sub_i32 s8, s3, s12                                      // 000000039938: 81880C03
	s_mul_i32 s6, s50, s10                                     // 00000003993C: 92060A32
	s_sub_u32 s13, s2, s6                                      // 000000039940: 808D0602
	s_cselect_b64 s[6:7], -1, 0                                // 000000039944: 858680C1
	s_cmp_lg_u64 s[6:7], 0                                     // 000000039948: BF138006
	s_subb_u32 s14, s8, s51                                    // 00000003994C: 828E3308
	s_sub_u32 s15, s13, s50                                    // 000000039950: 808F320D
	s_cselect_b64 s[8:9], -1, 0                                // 000000039954: 858880C1
	s_cmp_lg_u64 s[8:9], 0                                     // 000000039958: BF138008
	s_subb_u32 s8, s14, 0                                      // 00000003995C: 8288800E
	s_cmp_ge_u32 s8, s51                                       // 000000039960: BF093308
	s_cselect_b32 s9, -1, 0                                    // 000000039964: 850980C1
	s_cmp_ge_u32 s15, s50                                      // 000000039968: BF09320F
	s_cselect_b32 s14, -1, 0                                   // 00000003996C: 850E80C1
	s_cmp_eq_u32 s8, s51                                       // 000000039970: BF063308
	s_cselect_b32 s8, s14, s9                                  // 000000039974: 8508090E
	s_add_u32 s9, s10, 1                                       // 000000039978: 8009810A
	s_addc_u32 s14, s11, 0                                     // 00000003997C: 820E800B
	s_add_u32 s15, s10, 2                                      // 000000039980: 800F820A
	s_addc_u32 s16, s11, 0                                     // 000000039984: 8210800B
	s_cmp_lg_u32 s8, 0                                         // 000000039988: BF078008
	s_cselect_b32 s8, s15, s9                                  // 00000003998C: 8508090F
	s_cselect_b32 s9, s16, s14                                 // 000000039990: 85090E10
	s_cmp_lg_u64 s[6:7], 0                                     // 000000039994: BF138006
	s_subb_u32 s3, s3, s12                                     // 000000039998: 82830C03
	s_cmp_ge_u32 s3, s51                                       // 00000003999C: BF093303
	s_cselect_b32 s6, -1, 0                                    // 0000000399A0: 850680C1
	s_cmp_ge_u32 s13, s50                                      // 0000000399A4: BF09320D
	s_cselect_b32 s7, -1, 0                                    // 0000000399A8: 850780C1
	s_cmp_eq_u32 s3, s51                                       // 0000000399AC: BF063303
	s_cselect_b32 s3, s7, s6                                   // 0000000399B0: 85030607
	s_cmp_lg_u32 s3, 0                                         // 0000000399B4: BF078003
	s_cselect_b32 s65, s9, s11                                 // 0000000399B8: 85410B09
	s_cselect_b32 s64, s8, s10                                 // 0000000399BC: 85400A08
	s_cbranch_execnz 24                                        // 0000000399C0: BF890018 <EpCombineIntraNodeKernel_bf16_nop2p+0xa24>
	v_cvt_f32_u32_e32 v1, s50                                  // 0000000399C4: 7E020C32
	s_sub_i32 s3, 0, s50                                       // 0000000399C8: 81833280
	s_mov_b32 s65, 0                                           // 0000000399CC: BEC10080
	v_rcp_iflag_f32_e32 v1, v1                                 // 0000000399D0: 7E024701
	s_nop 0                                                    // 0000000399D4: BF800000
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 0000000399D8: 0A0202FF 4F7FFFFE
	v_cvt_u32_f32_e32 v1, v1                                   // 0000000399E0: 7E020F01
	s_nop 0                                                    // 0000000399E4: BF800000
	v_readfirstlane_b32 s4, v1                                 // 0000000399E8: 7E080501
	s_mul_i32 s3, s3, s4                                       // 0000000399EC: 92030403
	s_mul_hi_u32 s3, s4, s3                                    // 0000000399F0: 96030304
	s_add_i32 s4, s4, s3                                       // 0000000399F4: 81040304
	s_mul_hi_u32 s3, s2, s4                                    // 0000000399F8: 96030402
	s_mul_i32 s5, s3, s50                                      // 0000000399FC: 92053203
	s_sub_i32 s2, s2, s5                                       // 000000039A00: 81820502
	s_add_i32 s4, s3, 1                                        // 000000039A04: 81048103
	s_sub_i32 s5, s2, s50                                      // 000000039A08: 81853202
	s_cmp_ge_u32 s2, s50                                       // 000000039A0C: BF093202
	s_cselect_b32 s3, s4, s3                                   // 000000039A10: 85030304
	s_cselect_b32 s2, s5, s2                                   // 000000039A14: 85020205
	s_add_i32 s4, s3, 1                                        // 000000039A18: 81048103
	s_cmp_ge_u32 s2, s50                                       // 000000039A1C: BF093202
	s_cselect_b32 s64, s4, s3                                  // 000000039A20: 85400304
	s_cmp_gt_i32 s46, 63                                       // 000000039A24: BF02BF2E
	s_mov_b64 s[2:3], -1                                       // 000000039A28: BE8201C1
	s_cbranch_scc0 35                                          // 000000039A2C: BF840023 <EpCombineIntraNodeKernel_bf16_nop2p+0xabc>
	s_add_u32 s8, s0, 0x2e8                                    // 000000039A30: 8008FF00 000002E8
	s_addc_u32 s9, s1, 0                                       // 000000039A38: 82098001
	s_getpc_b64 s[0:1]                                         // 000000039A3C: BE801C00
	s_add_u32 s0, s0, 0xfffd3d6f                               // 000000039A40: 8000FF00 FFFD3D6F
	s_addc_u32 s1, s1, -1                                      // 000000039A48: 8201FF01 FFFFFFFF
	s_getpc_b64 s[2:3]                                         // 000000039A50: BE821C00
	s_add_u32 s2, s2, 0xfffd4d39                               // 000000039A54: 8002FF02 FFFD4D39
	s_addc_u32 s3, s3, -1                                      // 000000039A5C: 8203FF03 FFFFFFFF
	s_getpc_b64 s[4:5]                                         // 000000039A64: BE841C00
	s_add_u32 s4, s4, 0xfffd462a                               // 000000039A68: 8004FF04 FFFD462A
	s_addc_u32 s5, s5, -1                                      // 000000039A70: 8205FF05 FFFFFFFF
	s_getpc_b64 s[6:7]                                         // 000000039A78: BE861C00
	s_add_u32 s6, s6, 0xfffd7538                               // 000000039A7C: 8006FF06 FFFD7538
	s_addc_u32 s7, s7, -1                                      // 000000039A84: 8207FF07 FFFFFFFF
	v_mov_b32_e32 v43, v0                                      // 000000039A8C: 7E560300
	v_mov_b32_e32 v0, s0                                       // 000000039A90: 7E000200
	v_mov_b32_e32 v1, s1                                       // 000000039A94: 7E020201
	v_mov_b32_e32 v2, s2                                       // 000000039A98: 7E040202
	v_mov_b32_e32 v3, s3                                       // 000000039A9C: 7E060203
	v_mov_b32_e32 v4, 0x1c5                                    // 000000039AA0: 7E0802FF 000001C5
	v_mov_b32_e32 v5, s4                                       // 000000039AA8: 7E0A0204
	v_mov_b32_e32 v6, s5                                       // 000000039AAC: 7E0C0205
	s_swappc_b64 s[30:31], s[6:7]                              // 000000039AB0: BE9E1E06
	v_mov_b32_e32 v0, v43                                      // 000000039AB4: 7E00032B
	s_mov_b64 s[2:3], 0                                        // 000000039AB8: BE820180
	s_andn2_b64 vcc, exec, s[2:3]                              // 000000039ABC: 89EA027E
	s_cbranch_vccnz 3428                                       // 000000039AC0: BF870D64 <EpCombineIntraNodeKernel_bf16_nop2p+0x4054>
	s_mul_i32 s33, s50, s21                                    // 000000039AC4: 92211532
	v_cmp_gt_i32_e32 vcc, s33, v53                             // 000000039AC8: 7D886A21
	s_and_saveexec_b64 s[0:1], vcc                             // 000000039ACC: BE80206A
	s_cbranch_execz 3424                                       // 000000039AD0: BF880D60 <EpCombineIntraNodeKernel_bf16_nop2p+0x4054>
	v_mov_b32_e32 v21, 0                                       // 000000039AD4: 7E2A0280
	global_load_dwordx2 v[22:23], v21, s[22:23]                // 000000039AD8: DC548000 16160015
	s_mul_i32 s0, s58, s46                                     // 000000039AE0: 92002E3A
	s_lshl_b32 s0, s0, 3                                       // 000000039AE4: 8E008300
	s_add_i32 s0, s0, 0                                        // 000000039AE8: 81008000
	s_ashr_i32 s5, s52, 31                                     // 000000039AEC: 90059F34
	s_cmp_lt_i32 s53, 5                                        // 000000039AF0: BF048535
	s_cselect_b64 s[68:69], -1, 0                              // 000000039AF4: 85C480C1
	s_add_i32 s51, s50, -1                                     // 000000039AF8: 8133C132
	s_lshr_b64 s[70:71], s[46:47], 6                           // 000000039AFC: 8FC6862E
	v_readlane_b32 s2, v63, 6                                  // 000000039B00: D2890002 00010D3F
	s_cmp_gt_u32 s46, 63                                       // 000000039B08: BF08BF2E
	s_mul_i32 s6, s2, s53                                      // 000000039B0C: 92063502
	s_cselect_b64 s[2:3], -1, 0                                // 000000039B10: 858280C1
	v_writelane_b32 v63, s2, 7                                 // 000000039B14: D28A003F 00010E02
	s_and_b32 s66, s46, 7                                      // 000000039B1C: 8642872E
	s_and_b32 s74, s46, -8                                     // 000000039B20: 864AC82E
	v_writelane_b32 v63, s3, 8                                 // 000000039B24: D28A003F 00011003
	s_and_b32 s2, s46, 7                                       // 000000039B2C: 8602872E
	s_cmp_lg_u32 s2, 0                                         // 000000039B30: BF078002
	v_mul_lo_u32 v1, s46, v41                                  // 000000039B34: D2850001 0002522E
	s_mov_b32 s4, s52                                          // 000000039B3C: BE840034
	v_lshlrev_b32_e32 v0, 1, v0                                // 000000039B40: 24000081
	s_cselect_b64 s[76:77], -1, 0                              // 000000039B44: 85CC80C1
	s_abs_i32 s52, s50                                         // 000000039B48: BEB43032
	v_cmp_gt_u32_e64 s[2:3], s46, v42                          // 000000039B4C: D0CC0002 0002542E
	v_lshlrev_b32_e32 v1, 3, v1                                // 000000039B54: 24020283
	v_and_b32_e32 v26, 0x7e, v0                                // 000000039B58: 263400FF 0000007E
	v_cvt_f32_u32_e32 v0, s52                                  // 000000039B60: 7E000C34
	v_writelane_b32 v63, s2, 9                                 // 000000039B64: D28A003F 00011202
	v_add_u32_e32 v27, 0, v1                                   // 000000039B6C: 68360280
	v_add_u32_e32 v29, s0, v1                                  // 000000039B70: 683A0200
	v_lshlrev_b32_e32 v1, 3, v42                               // 000000039B74: 24025483
	v_writelane_b32 v63, s3, 10                                // 000000039B78: D28A003F 00011403
	s_abs_i32 s2, s6                                           // 000000039B80: BE823006
	v_add_u32_e32 v48, v27, v1                                 // 000000039B84: 6860031B
	v_add_u32_e32 v49, v29, v1                                 // 000000039B88: 6862031D
	v_cvt_f32_u32_e32 v1, s2                                   // 000000039B8C: 7E020C02
	v_rcp_iflag_f32_e32 v0, v0                                 // 000000039B90: 7E004700
	v_lshlrev_b64 v[2:3], v42, -1                              // 000000039B94: D28F0002 0001832A
	s_sub_i32 s7, 0, s52                                       // 000000039B9C: 81873480
	v_rcp_iflag_f32_e32 v1, v1                                 // 000000039BA0: 7E024701
	v_mul_f32_e32 v0, 0x4f7ffffe, v0                           // 000000039BA4: 0A0000FF 4F7FFFFE
	v_cvt_u32_f32_e32 v0, v0                                   // 000000039BAC: 7E000F00
	s_mov_b32 s67, 0                                           // 000000039BB0: BEC30080
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000039BB4: 0A0202FF 4F7FFFFE
	v_cvt_u32_f32_e32 v1, v1                                   // 000000039BBC: 7E020F01
	v_not_b32_e32 v24, v2                                      // 000000039BC0: 7E305702
	v_mul_lo_u32 v2, s7, v0                                    // 000000039BC4: D2850002 00020007
	s_mov_b64 s[78:79], s[66:67]                               // 000000039BCC: BECE0142
	v_mul_hi_u32 v2, v0, v2                                    // 000000039BD0: D2860002 00020500
	s_ashr_i32 s66, s6, 31                                     // 000000039BD8: 90429F06
	s_sub_i32 s6, 0, s2                                        // 000000039BDC: 81860280
	v_add_u32_e32 v50, v0, v2                                  // 000000039BE0: 68640500
	v_mul_lo_u32 v0, s6, v1                                    // 000000039BE4: D2850000 00020206
	v_mul_hi_u32 v0, v1, v0                                    // 000000039BEC: D2860000 00020101
	s_lshl_b32 s6, s58, 3                                      // 000000039BF4: 8E06833A
	v_add_u32_e32 v51, v1, v0                                  // 000000039BF8: 68660101
	v_lshl_add_u32 v0, v41, 3, s6                              // 000000039BFC: D1FD0000 00190729
	s_lshl_b32 s6, s46, 3                                      // 000000039C04: 8E06832E
	s_andn2_b32 s6, s6, 63                                     // 000000039C08: 8906BF06
	v_mul_lo_u32 v0, s46, v0                                   // 000000039C0C: D2850000 0002002E
	s_add_i32 s6, s6, 0                                        // 000000039C14: 81068006
	s_lshl_b64 s[4:5], s[4:5], 3                               // 000000039C18: 8E848304
	v_cmp_gt_i32_e64 s[0:1], s46, v42                          // 000000039C1C: D0C40000 0002542E
	v_mov_b32_e32 v43, v21                                     // 000000039C24: 7E560315
	v_not_b32_e32 v25, v3                                      // 000000039C28: 7E325703
	v_lshlrev_b32_e32 v28, 1, v42                              // 000000039C2C: 24385481
	s_mov_b32 s75, s47                                         // 000000039C30: BECB002F
	s_ashr_i32 s3, s50, 31                                     // 000000039C34: 90039F32
	v_lshlrev_b32_e32 v30, 2, v42                              // 000000039C38: 243C5482
	v_mov_b32_e32 v31, v21                                     // 000000039C3C: 7E3E0315
	v_add_u32_e32 v52, s6, v0                                  // 000000039C40: 68680006
	s_mov_b64 s[80:81], 0                                      // 000000039C44: BED00180
	v_writelane_b32 v63, s4, 11                                // 000000039C48: D28A003F 00011604
	s_mov_b64 s[84:85], 0x7f                                   // 000000039C50: BED401FF 0000007F
	s_mov_b32 s82, 0xffff0000                                  // 000000039C58: BED200FF FFFF0000
	s_mov_b32 s58, 0x7f800000                                  // 000000039C60: BEBA00FF 7F800000
	s_movk_i32 s59, 0x7fff                                     // 000000039C68: B03B7FFF
	s_mov_b64 s[86:87], 0x80                                   // 000000039C6C: BED601FF 00000080
	s_mov_b64 s[88:89], 0x100                                  // 000000039C74: BED801FF 00000100
	v_writelane_b32 v63, s5, 12                                // 000000039C7C: D28A003F 00011805
	s_branch 6                                                 // 000000039C84: BF820006 <EpCombineIntraNodeKernel_bf16_nop2p+0xca0>
	s_or_b64 exec, exec, s[24:25]                              // 000000039C88: 87FE187E
	v_add_u32_e32 v53, s83, v53                                // 000000039C8C: 686A6A53
	v_cmp_le_i32_e32 vcc, s33, v53                             // 000000039C90: 7D866A21
	s_or_b64 s[80:81], vcc, s[80:81]                           // 000000039C94: 87D0506A
	s_andn2_b64 exec, exec, s[80:81]                           // 000000039C98: 89FE507E
	s_cbranch_execz 3309                                       // 000000039C9C: BF880CED <EpCombineIntraNodeKernel_bf16_nop2p+0x4054>
	v_sub_u32_e32 v1, 0, v53                                   // 000000039CA0: 6A026A80
	v_max_i32_e32 v1, v53, v1                                  // 000000039CA4: 1A020335
	v_mul_hi_u32 v2, v1, v50                                   // 000000039CA8: D2860002 00026501
	v_mul_lo_u32 v3, v2, s52                                   // 000000039CB0: D2850003 00006902
	v_sub_u32_e32 v1, v1, v3                                   // 000000039CB8: 6A020701
	v_add_u32_e32 v3, 1, v2                                    // 000000039CBC: 68060481
	v_cmp_le_u32_e32 vcc, s52, v1                              // 000000039CC0: 7D960234
	v_ashrrev_i32_e32 v0, 31, v53                              // 000000039CC4: 22006A9F
	v_xor_b32_e32 v0, s3, v0                                   // 000000039CC8: 2A000003
	v_cndmask_b32_e32 v2, v2, v3, vcc                          // 000000039CCC: 00040702
	v_subrev_u32_e32 v3, s52, v1                               // 000000039CD0: 6C060234
	v_cndmask_b32_e32 v1, v1, v3, vcc                          // 000000039CD4: 00020701
	v_add_u32_e32 v3, 1, v2                                    // 000000039CD8: 68060481
	v_cmp_le_u32_e32 vcc, s52, v1                              // 000000039CDC: 7D960234
	s_nop 1                                                    // 000000039CE0: BF800001
	v_cndmask_b32_e32 v1, v2, v3, vcc                          // 000000039CE4: 00020702
	v_xor_b32_e32 v1, v1, v0                                   // 000000039CE8: 2A020101
	v_sub_u32_e32 v32, v1, v0                                  // 000000039CEC: 6A400101
	v_mul_lo_u32 v0, v32, s50                                  // 000000039CF0: D2850000 00006520
	v_sub_u32_e32 v33, v53, v0                                 // 000000039CF8: 6A420135
	v_ashrrev_i32_e32 v0, 31, v33                              // 000000039CFC: 2200429F
	v_mul_lo_u32 v2, s64, v0                                   // 000000039D00: D2850002 00020040
	v_mul_lo_u32 v3, s65, v33                                  // 000000039D08: D2850003 00024241
	v_mad_u64_u32 v[0:1], s[4:5], s64, v33, 0                  // 000000039D10: D1E80400 02024240
	v_add3_u32 v1, v1, v2, v3                                  // 000000039D18: D1FF0001 040E0501
	s_and_saveexec_b64 s[4:5], s[0:1]                          // 000000039D20: BE842000
	s_cbranch_execz 77                                         // 000000039D24: BF88004D <EpCombineIntraNodeKernel_bf16_nop2p+0xe5c>
	v_mad_u64_u32 v[2:3], s[6:7], v32, s46, v[42:43]           // 000000039D28: D1E80602 04A85D20
	v_readlane_b32 s8, v63, 0                                  // 000000039D30: D2890008 0001013F
	v_ashrrev_i32_e32 v3, 31, v2                               // 000000039D38: 2206049F
	v_readlane_b32 s9, v63, 1                                  // 000000039D3C: D2890009 0001033F
	v_readlane_b32 s10, v63, 2                                 // 000000039D44: D289000A 0001053F
	v_readlane_b32 s11, v63, 3                                 // 000000039D4C: D289000B 0001073F
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[8:9]                   // 000000039D54: D2080002 00210502
	global_load_dword v4, v[2:3], off                          // 000000039D5C: DC508000 047F0002
	v_mov_b64_e32 v[2:3], 0                                    // 000000039D64: 7E047080
	s_waitcnt vmcnt(0)                                         // 000000039D68: BF8C0F70
	v_sub_u32_e32 v6, 0, v4                                    // 000000039D6C: 6A0C0880
	v_ashrrev_i32_e32 v5, 31, v4                               // 000000039D70: 220A089F
	v_max_i32_e32 v4, v4, v6                                   // 000000039D74: 1A080D04
	v_mul_hi_u32 v6, v4, v51                                   // 000000039D78: D2860006 00026704
	v_mul_lo_u32 v7, v6, s2                                    // 000000039D80: D2850007 00000506
	v_sub_u32_e32 v4, v4, v7                                   // 000000039D88: 6A080F04
	v_add_u32_e32 v8, 1, v6                                    // 000000039D8C: 68100C81
	v_cmp_le_u32_e32 vcc, s2, v4                               // 000000039D90: 7D960802
	v_subrev_u32_e32 v7, s2, v4                                // 000000039D94: 6C0E0802
	v_xor_b32_e32 v5, s66, v5                                  // 000000039D98: 2A0A0A42
	v_cndmask_b32_e32 v6, v6, v8, vcc                          // 000000039D9C: 000C1106
	v_cndmask_b32_e32 v4, v4, v7, vcc                          // 000000039DA0: 00080F04
	v_add_u32_e32 v7, 1, v6                                    // 000000039DA4: 680E0C81
	v_cmp_le_u32_e32 vcc, s2, v4                               // 000000039DA8: 7D960802
	s_nop 1                                                    // 000000039DAC: BF800001
	v_cndmask_b32_e32 v4, v6, v7, vcc                          // 000000039DB0: 00080F06
	v_xor_b32_e32 v4, v4, v5                                   // 000000039DB4: 2A080B04
	v_sub_u32_e32 v6, v4, v5                                   // 000000039DB8: 6A0C0B04
	v_cmp_gt_i32_e32 vcc, s53, v6                              // 000000039DBC: 7D880C35
	v_mov_b64_e32 v[4:5], 0                                    // 000000039DC0: 7E087080
	s_and_saveexec_b64 s[6:7], vcc                             // 000000039DC4: BE86206A
	s_cbranch_execz 31                                         // 000000039DC8: BF88001F <EpCombineIntraNodeKernel_bf16_nop2p+0xe48>
	global_load_dwordx2 v[2:3], v21, s[48:49] offset:16        // 000000039DCC: DC548010 02300015
	v_readlane_b32 s8, v63, 11                                 // 000000039DD4: D2890008 0001173F
	v_readlane_b32 s9, v63, 12                                 // 000000039DDC: D2890009 0001193F
	s_waitcnt vmcnt(0)                                         // 000000039DE4: BF8C0F70
	s_nop 0                                                    // 000000039DE8: BF800000
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[8:9]                   // 000000039DEC: D2080002 00210102
	flat_load_dwordx2 v[2:3], v[2:3]                           // 000000039DF4: DC540000 02000002
	v_readlane_b32 s8, v63, 6                                  // 000000039DFC: D2890008 00010D3F
	s_nop 1                                                    // 000000039E04: BF800001
	v_mad_u64_u32 v[4:5], s[8:9], v6, s8, v[32:33]             // 000000039E08: D1E80804 04801106
	v_ashrrev_i32_e32 v5, 31, v4                               // 000000039E10: 220A089F
	v_mul_lo_u32 v6, s57, v4                                   // 000000039E14: D2850006 00020839
	v_mul_lo_u32 v7, s56, v5                                   // 000000039E1C: D2850007 00020A38
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 000000039E24: BF8C0070
	v_mad_u64_u32 v[4:5], s[8:9], s56, v4, v[2:3]              // 000000039E28: D1E80804 040A0838
	v_add3_u32 v5, v6, v5, v7                                  // 000000039E30: D1FF0005 041E0B06
	v_lshl_add_u64 v[2:3], v[0:1], 1, v[4:5]                   // 000000039E38: D2080002 04110300
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[60:61]                 // 000000039E40: D2080004 00F10104
	s_or_b64 exec, exec, s[6:7]                                // 000000039E48: 87FE067E
	ds_write_b64 v48, v[2:3]                                   // 000000039E4C: D89A0000 00000230
	ds_write_b64 v49, v[4:5]                                   // 000000039E54: D89A0000 00000431
	s_or_b64 exec, exec, s[4:5]                                // 000000039E5C: 87FE047E
	s_and_b64 vcc, exec, s[68:69]                              // 000000039E60: 86EA447E
	s_mov_b32 s90, s46                                         // 000000039E64: BEDA002E
	s_cbranch_vccz 30                                          // 000000039E68: BF86001E <EpCombineIntraNodeKernel_bf16_nop2p+0xee4>
	v_mov_b64_e32 v[2:3], 0                                    // 000000039E6C: 7E047080
	s_mov_b64 s[4:5], 0                                        // 000000039E70: BE840180
	s_and_saveexec_b64 s[6:7], s[0:1]                          // 000000039E74: BE862000
	s_cbranch_execz 5                                          // 000000039E78: BF880005 <EpCombineIntraNodeKernel_bf16_nop2p+0xe90>
	ds_read_b64 v[2:3], v48                                    // 000000039E7C: D8EC0000 02000030
	s_waitcnt lgkmcnt(0)                                       // 000000039E84: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[2:3]                            // 000000039E88: 7DDA0480
	s_and_b64 s[4:5], vcc, exec                                // 000000039E8C: 86847E6A
	s_or_b64 exec, exec, s[6:7]                                // 000000039E90: 87FE067E
	v_cndmask_b32_e64 v4, 0, 1, s[4:5]                         // 000000039E94: D1000004 00110280
	v_cmp_ne_u32_e32 vcc, 0, v4                                // 000000039E9C: 7D9A0880
	s_bcnt1_i32_b64 s90, vcc                                   // 000000039EA0: BEDA0D6A
	s_cmp_gt_i32 s46, s90                                      // 000000039EA4: BF025A2E
	s_cselect_b64 s[6:7], -1, 0                                // 000000039EA8: 858680C1
	s_and_b64 s[6:7], s[4:5], s[6:7]                           // 000000039EAC: 86860604
	s_and_saveexec_b64 s[4:5], s[6:7]                          // 000000039EB0: BE842006
	s_cbranch_execz 10                                         // 000000039EB4: BF88000A <EpCombineIntraNodeKernel_bf16_nop2p+0xee0>
	v_and_b32_e32 v5, vcc_lo, v24                              // 000000039EB8: 260A306A
	v_and_b32_e32 v4, vcc_hi, v25                              // 000000039EBC: 2608326B
	v_bcnt_u32_b32 v5, v5, 0                                   // 000000039EC0: D28B0005 00010105
	v_bcnt_u32_b32 v4, v4, v5                                  // 000000039EC8: D28B0004 00020B04
	v_lshl_add_u32 v4, v4, 3, v27                              // 000000039ED0: D1FD0004 046D0704
	ds_write_b64 v4, v[2:3]                                    // 000000039ED8: D89A0000 00000204
	s_or_b64 exec, exec, s[4:5]                                // 000000039EE0: 87FE047E
	v_mad_i64_i32 v[2:3], s[4:5], v32, s54, 0                  // 000000039EE4: D1E90402 02006D20
	s_waitcnt vmcnt(0)                                         // 000000039EEC: BF8C0F70
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[22:23]                 // 000000039EF0: D2080002 04590302
	v_lshl_add_u64 v[34:35], v[0:1], 1, v[2:3]                 // 000000039EF8: D2080022 04090300
	v_mov_b32_e32 v3, s55                                      // 000000039F00: 7E060237
	v_sub_co_u32_e32 v2, vcc, s54, v0                          // 000000039F04: 34040036
	v_mov_b32_e32 v4, s64                                      // 000000039F08: 7E080240
	s_nop 0                                                    // 000000039F0C: BF800000
	v_subb_co_u32_e32 v3, vcc, v3, v1, vcc                     // 000000039F10: 3A060303
	v_cmp_lt_u64_e32 vcc, s[64:65], v[2:3]                     // 000000039F14: 7DD20440
	s_mov_b64 s[6:7], -1                                       // 000000039F18: BE8601C1
	s_mov_b64 s[92:93], 0                                      // 000000039F1C: BEDC0180
	v_cndmask_b32_e32 v2, v2, v4, vcc                          // 000000039F20: 00040902
	v_mov_b32_e32 v4, s65                                      // 000000039F24: 7E080241
	v_cndmask_b32_e32 v3, v3, v4, vcc                          // 000000039F28: 00060903
	v_cmp_gt_u64_e32 vcc, s[54:55], v[0:1]                     // 000000039F2C: 7DD80036
	s_cmp_lt_i32 s90, 6                                        // 000000039F30: BF04865A
	s_mov_b64 s[4:5], 0                                        // 000000039F34: BE840180
	v_cndmask_b32_e32 v37, 0, v3, vcc                          // 000000039F38: 004A0680
	v_cndmask_b32_e32 v36, 0, v2, vcc                          // 000000039F3C: 00480480
	s_cbranch_scc1 1313                                        // 000000039F40: BF850521 <EpCombineIntraNodeKernel_bf16_nop2p+0x23c8>
	s_cmp_gt_i32 s90, 7                                        // 000000039F44: BF02875A
	s_cbranch_scc0 955                                         // 000000039F48: BF8403BB <EpCombineIntraNodeKernel_bf16_nop2p+0x1e38>
	s_cmp_gt_i32 s90, 9                                        // 000000039F4C: BF02895A
	s_cbranch_scc0 519                                         // 000000039F50: BF840207 <EpCombineIntraNodeKernel_bf16_nop2p+0x1770>
	s_cmp_eq_u32 s90, 10                                       // 000000039F54: BF068A5A
	s_mov_b64 s[4:5], -1                                       // 000000039F58: BE8401C1
	s_cbranch_scc0 515                                         // 000000039F5C: BF840203 <EpCombineIntraNodeKernel_bf16_nop2p+0x176c>
	v_cmp_lt_u64_e32 vcc, s[84:85], v[36:37]                   // 000000039F60: 7DD24854
	v_mov_b64_e32 v[44:45], 0                                  // 000000039F64: 7E587080
	s_and_saveexec_b64 s[94:95], vcc                           // 000000039F68: BEDE206A
	s_cbranch_execz 316                                        // 000000039F6C: BF88013C <EpCombineIntraNodeKernel_bf16_nop2p+0x1460>
	ds_read2_b64 v[0:3], v27 offset0:8 offset1:9               // 000000039F70: D8EE0908 0000001B
	ds_read2_b64 v[4:7], v27 offset0:2 offset1:3               // 000000039F78: D8EE0302 0400001B
	ds_read2_b64 v[8:11], v27 offset1:1                        // 000000039F80: D8EE0100 0800001B
	ds_read2_b64 v[12:15], v27 offset0:6 offset1:7             // 000000039F88: D8EE0706 0C00001B
	ds_read2_b64 v[16:19], v27 offset0:4 offset1:5             // 000000039F90: D8EE0504 1000001B
	v_lshlrev_b32_e32 v20, 1, v26                              // 000000039F98: 24283481
	v_writelane_b32 v63, s48, 13                               // 000000039F9C: D28A003F 00011A30
	v_lshl_add_u64 v[38:39], v[34:35], 0, v[20:21]             // 000000039FA4: D2080026 04510122
	v_lshrrev_b64 v[40:41], 7, v[36:37]                        // 000000039FAC: D2900028 00024887
	s_mov_b64 s[96:97], 0                                      // 000000039FB4: BEE00180
	s_waitcnt lgkmcnt(0)                                       // 000000039FB8: BF8CC07F
	v_cmp_eq_u64_e32 vcc, 0, v[8:9]                            // 000000039FBC: 7DD41080
	v_cmp_ne_u64_e64 s[4:5], 0, v[8:9]                         // 000000039FC0: D0ED0004 00021080
	v_cmp_eq_u64_e64 s[6:7], 0, v[10:11]                       // 000000039FC8: D0EA0006 00021480
	v_cmp_ne_u64_e64 s[8:9], 0, v[10:11]                       // 000000039FD0: D0ED0008 00021480
	v_cmp_eq_u64_e64 s[10:11], 0, v[4:5]                       // 000000039FD8: D0EA000A 00020880
	v_cmp_ne_u64_e64 s[12:13], 0, v[4:5]                       // 000000039FE0: D0ED000C 00020880
	v_cmp_eq_u64_e64 s[14:15], 0, v[6:7]                       // 000000039FE8: D0EA000E 00020C80
	v_cmp_ne_u64_e64 s[16:17], 0, v[6:7]                       // 000000039FF0: D0ED0010 00020C80
	s_waitcnt lgkmcnt(0)                                       // 000000039FF8: BF8CC07F
	v_cmp_eq_u64_e64 s[18:19], 0, v[16:17]                     // 000000039FFC: D0EA0012 00022080
	v_cmp_ne_u64_e64 s[20:21], 0, v[16:17]                     // 00000003A004: D0ED0014 00022080
	v_cmp_eq_u64_e64 s[22:23], 0, v[18:19]                     // 00000003A00C: D0EA0016 00022480
	v_cmp_ne_u64_e64 s[24:25], 0, v[18:19]                     // 00000003A014: D0ED0018 00022480
	v_cmp_eq_u64_e64 s[26:27], 0, v[12:13]                     // 00000003A01C: D0EA001A 00021880
	v_cmp_ne_u64_e64 s[28:29], 0, v[12:13]                     // 00000003A024: D0ED001C 00021880
	v_cmp_eq_u64_e64 s[30:31], 0, v[14:15]                     // 00000003A02C: D0EA001E 00021C80
	v_cmp_ne_u64_e64 s[34:35], 0, v[14:15]                     // 00000003A034: D0ED0022 00021C80
	v_cmp_eq_u64_e64 s[36:37], 0, v[0:1]                       // 00000003A03C: D0EA0024 00020080
	v_cmp_ne_u64_e64 s[38:39], 0, v[0:1]                       // 00000003A044: D0ED0026 00020080
	v_cmp_eq_u64_e64 s[40:41], 0, v[2:3]                       // 00000003A04C: D0EA0028 00020480
	v_cmp_ne_u64_e64 s[42:43], 0, v[2:3]                       // 00000003A054: D0ED002A 00020480
	s_mov_b64 s[98:99], 0                                      // 00000003A05C: BEE20180
	v_writelane_b32 v63, s49, 14                               // 00000003A060: D28A003F 00011C31
	s_branch 39                                                // 00000003A068: BF820027 <EpCombineIntraNodeKernel_bf16_nop2p+0x1108>
	s_or_b64 exec, exec, s[72:73]                              // 00000003A06C: 87FE487E
	v_lshrrev_b32_e32 v47, 16, v44                             // 00000003A070: 205E5890
	v_lshl_add_u64 v[44:45], s[98:99], 1, v[38:39]             // 00000003A074: D208002C 04990262
	s_add_u32 s98, s98, 0x80                                   // 00000003A07C: 8062FF62 00000080
	v_lshl_add_u64 v[40:41], v[40:41], 0, -1                   // 00000003A084: D2080028 03050128
	v_and_or_b32 v46, v46, s82, v47                            // 00000003A08C: D201002E 04BCA52E
	s_addc_u32 s99, s99, 0                                     // 00000003A094: 82638063
	v_cmp_eq_u64_e64 s[44:45], 0, v[40:41]                     // 00000003A098: D0EA002C 00025080
	flat_store_dword v[44:45], v46 nt                          // 00000003A0A0: DC720000 00002E2C
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[88:89]                 // 00000003A0A8: D2080002 01610102
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[88:89]                 // 00000003A0B0: D2080000 01610100
	v_lshl_add_u64 v[14:15], v[14:15], 0, s[88:89]             // 00000003A0B8: D208000E 0161010E
	v_lshl_add_u64 v[12:13], v[12:13], 0, s[88:89]             // 00000003A0C0: D208000C 0161010C
	v_lshl_add_u64 v[18:19], v[18:19], 0, s[88:89]             // 00000003A0C8: D2080012 01610112
	v_lshl_add_u64 v[16:17], v[16:17], 0, s[88:89]             // 00000003A0D0: D2080010 01610110
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[88:89]                 // 00000003A0D8: D2080006 01610106
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[88:89]                 // 00000003A0E0: D2080004 01610104
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[88:89]             // 00000003A0E8: D208000A 0161010A
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[88:89]                 // 00000003A0F0: D2080008 01610108
	s_or_b64 s[96:97], s[44:45], s[96:97]                      // 00000003A0F8: 87E0602C
	v_mov_b64_e32 v[44:45], s[98:99]                           // 00000003A0FC: 7E587062
	s_andn2_b64 exec, exec, s[96:97]                           // 00000003A100: 89FE607E
	s_cbranch_execz 209                                        // 00000003A104: BF8800D1 <EpCombineIntraNodeKernel_bf16_nop2p+0x144c>
	s_and_saveexec_b64 s[44:45], s[4:5]                        // 00000003A108: BEAC2004
	s_cbranch_execz 4                                          // 00000003A10C: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x1120>
	v_lshl_add_u64 v[44:45], v[8:9], 0, v[30:31]               // 00000003A110: D208002C 04790108
	flat_load_dword v60, v[44:45] nt                           // 00000003A118: DC520000 3C00002C
	s_or_b64 exec, exec, s[44:45]                              // 00000003A120: 87FE2C7E
	s_and_saveexec_b64 s[44:45], s[8:9]                        // 00000003A124: BEAC2008
	s_cbranch_execz 4                                          // 00000003A128: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x113c>
	v_lshl_add_u64 v[44:45], v[10:11], 0, v[30:31]             // 00000003A12C: D208002C 0479010A
	flat_load_dword v59, v[44:45] nt                           // 00000003A134: DC520000 3B00002C
	s_or_b64 exec, exec, s[44:45]                              // 00000003A13C: 87FE2C7E
	s_and_saveexec_b64 s[44:45], s[12:13]                      // 00000003A140: BEAC200C
	s_cbranch_execz 4                                          // 00000003A144: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x1158>
	v_lshl_add_u64 v[44:45], v[4:5], 0, v[30:31]               // 00000003A148: D208002C 04790104
	flat_load_dword v58, v[44:45] nt                           // 00000003A150: DC520000 3A00002C
	s_or_b64 exec, exec, s[44:45]                              // 00000003A158: 87FE2C7E
	s_and_saveexec_b64 s[44:45], s[16:17]                      // 00000003A15C: BEAC2010
	s_cbranch_execz 4                                          // 00000003A160: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x1174>
	v_lshl_add_u64 v[44:45], v[6:7], 0, v[30:31]               // 00000003A164: D208002C 04790106
	flat_load_dword v57, v[44:45] nt                           // 00000003A16C: DC520000 3900002C
	s_or_b64 exec, exec, s[44:45]                              // 00000003A174: 87FE2C7E
	s_and_saveexec_b64 s[44:45], s[20:21]                      // 00000003A178: BEAC2014
	s_cbranch_execz 4                                          // 00000003A17C: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x1190>
	v_lshl_add_u64 v[44:45], v[16:17], 0, v[30:31]             // 00000003A180: D208002C 04790110
	flat_load_dword v56, v[44:45] nt                           // 00000003A188: DC520000 3800002C
	s_or_b64 exec, exec, s[44:45]                              // 00000003A190: 87FE2C7E
	s_and_saveexec_b64 s[44:45], s[24:25]                      // 00000003A194: BEAC2018
	s_cbranch_execz 4                                          // 00000003A198: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x11ac>
	v_lshl_add_u64 v[44:45], v[18:19], 0, v[30:31]             // 00000003A19C: D208002C 04790112
	flat_load_dword v55, v[44:45] nt                           // 00000003A1A4: DC520000 3700002C
	s_or_b64 exec, exec, s[44:45]                              // 00000003A1AC: 87FE2C7E
	s_and_saveexec_b64 s[44:45], s[28:29]                      // 00000003A1B0: BEAC201C
	s_cbranch_execz 4                                          // 00000003A1B4: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x11c8>
	v_lshl_add_u64 v[44:45], v[12:13], 0, v[30:31]             // 00000003A1B8: D208002C 0479010C
	flat_load_dword v54, v[44:45] nt                           // 00000003A1C0: DC520000 3600002C
	s_or_b64 exec, exec, s[44:45]                              // 00000003A1C8: 87FE2C7E
	s_and_saveexec_b64 s[44:45], s[34:35]                      // 00000003A1CC: BEAC2022
	s_cbranch_execz 4                                          // 00000003A1D0: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x11e4>
	v_lshl_add_u64 v[44:45], v[14:15], 0, v[30:31]             // 00000003A1D4: D208002C 0479010E
	flat_load_dword v20, v[44:45] nt                           // 00000003A1DC: DC520000 1400002C
	s_or_b64 exec, exec, s[44:45]                              // 00000003A1E4: 87FE2C7E
	s_and_saveexec_b64 s[44:45], s[38:39]                      // 00000003A1E8: BEAC2026
	s_cbranch_execz 4                                          // 00000003A1EC: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x1200>
	v_lshl_add_u64 v[44:45], v[0:1], 0, v[30:31]               // 00000003A1F0: D208002C 04790100
	flat_load_dword v61, v[44:45] nt                           // 00000003A1F8: DC520000 3D00002C
	s_or_b64 exec, exec, s[44:45]                              // 00000003A200: 87FE2C7E
	s_and_saveexec_b64 s[44:45], s[42:43]                      // 00000003A204: BEAC202A
	s_cbranch_execz 4                                          // 00000003A208: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x121c>
	v_lshl_add_u64 v[44:45], v[2:3], 0, v[30:31]               // 00000003A20C: D208002C 04790102
	flat_load_dword v62, v[44:45] nt                           // 00000003A214: DC520000 3E00002C
	s_or_b64 exec, exec, s[44:45]                              // 00000003A21C: 87FE2C7E
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003A220: BF8C0070
	v_lshlrev_b32_e32 v44, 16, v60                             // 00000003A224: 24587890
	v_and_b32_e32 v45, 0xffff0000, v60                         // 00000003A228: 265A78FF FFFF0000
	v_pk_add_f32 v[44:45], v[44:45], 0 op_sel_hi:[1,0]         // 00000003A230: D3B2402C 0801012C
	v_lshlrev_b32_e32 v46, 16, v59                             // 00000003A238: 245C7690
	v_cndmask_b32_e64 v45, v45, 0, vcc                         // 00000003A23C: D100002D 01A9012D
	v_cndmask_b32_e64 v44, v44, 0, vcc                         // 00000003A244: D100002C 01A9012C
	v_and_b32_e32 v47, 0xffff0000, v59                         // 00000003A24C: 265E76FF FFFF0000
	v_pk_add_f32 v[46:47], v[44:45], v[46:47]                  // 00000003A254: D3B2402E 18025D2C
	s_nop 0                                                    // 00000003A25C: BF800000
	v_cndmask_b32_e64 v45, v47, v45, s[6:7]                    // 00000003A260: D100002D 001A5B2F
	v_cndmask_b32_e64 v44, v46, v44, s[6:7]                    // 00000003A268: D100002C 001A592E
	v_lshlrev_b32_e32 v46, 16, v58                             // 00000003A270: 245C7490
	v_and_b32_e32 v47, 0xffff0000, v58                         // 00000003A274: 265E74FF FFFF0000
	v_pk_add_f32 v[46:47], v[44:45], v[46:47]                  // 00000003A27C: D3B2402E 18025D2C
	s_nop 0                                                    // 00000003A284: BF800000
	v_cndmask_b32_e64 v45, v47, v45, s[10:11]                  // 00000003A288: D100002D 002A5B2F
	v_cndmask_b32_e64 v44, v46, v44, s[10:11]                  // 00000003A290: D100002C 002A592E
	v_lshlrev_b32_e32 v46, 16, v57                             // 00000003A298: 245C7290
	v_and_b32_e32 v47, 0xffff0000, v57                         // 00000003A29C: 265E72FF FFFF0000
	v_pk_add_f32 v[46:47], v[44:45], v[46:47]                  // 00000003A2A4: D3B2402E 18025D2C
	s_nop 0                                                    // 00000003A2AC: BF800000
	v_cndmask_b32_e64 v45, v47, v45, s[14:15]                  // 00000003A2B0: D100002D 003A5B2F
	v_cndmask_b32_e64 v44, v46, v44, s[14:15]                  // 00000003A2B8: D100002C 003A592E
	v_lshlrev_b32_e32 v46, 16, v56                             // 00000003A2C0: 245C7090
	v_and_b32_e32 v47, 0xffff0000, v56                         // 00000003A2C4: 265E70FF FFFF0000
	v_pk_add_f32 v[46:47], v[44:45], v[46:47]                  // 00000003A2CC: D3B2402E 18025D2C
	s_nop 0                                                    // 00000003A2D4: BF800000
	v_cndmask_b32_e64 v45, v47, v45, s[18:19]                  // 00000003A2D8: D100002D 004A5B2F
	v_cndmask_b32_e64 v44, v46, v44, s[18:19]                  // 00000003A2E0: D100002C 004A592E
	v_lshlrev_b32_e32 v46, 16, v55                             // 00000003A2E8: 245C6E90
	v_and_b32_e32 v47, 0xffff0000, v55                         // 00000003A2EC: 265E6EFF FFFF0000
	v_pk_add_f32 v[46:47], v[44:45], v[46:47]                  // 00000003A2F4: D3B2402E 18025D2C
	s_nop 0                                                    // 00000003A2FC: BF800000
	v_cndmask_b32_e64 v45, v47, v45, s[22:23]                  // 00000003A300: D100002D 005A5B2F
	v_cndmask_b32_e64 v44, v46, v44, s[22:23]                  // 00000003A308: D100002C 005A592E
	v_lshlrev_b32_e32 v46, 16, v54                             // 00000003A310: 245C6C90
	v_and_b32_e32 v47, 0xffff0000, v54                         // 00000003A314: 265E6CFF FFFF0000
	v_pk_add_f32 v[46:47], v[44:45], v[46:47]                  // 00000003A31C: D3B2402E 18025D2C
	s_nop 0                                                    // 00000003A324: BF800000
	v_cndmask_b32_e64 v45, v47, v45, s[26:27]                  // 00000003A328: D100002D 006A5B2F
	v_cndmask_b32_e64 v44, v46, v44, s[26:27]                  // 00000003A330: D100002C 006A592E
	v_lshlrev_b32_e32 v46, 16, v20                             // 00000003A338: 245C2890
	v_and_b32_e32 v47, 0xffff0000, v20                         // 00000003A33C: 265E28FF FFFF0000
	v_pk_add_f32 v[46:47], v[44:45], v[46:47]                  // 00000003A344: D3B2402E 18025D2C
	s_nop 0                                                    // 00000003A34C: BF800000
	v_cndmask_b32_e64 v45, v47, v45, s[30:31]                  // 00000003A350: D100002D 007A5B2F
	v_cndmask_b32_e64 v44, v46, v44, s[30:31]                  // 00000003A358: D100002C 007A592E
	v_lshlrev_b32_e32 v46, 16, v61                             // 00000003A360: 245C7A90
	v_and_b32_e32 v47, 0xffff0000, v61                         // 00000003A364: 265E7AFF FFFF0000
	v_pk_add_f32 v[46:47], v[44:45], v[46:47]                  // 00000003A36C: D3B2402E 18025D2C
	s_nop 0                                                    // 00000003A374: BF800000
	v_cndmask_b32_e64 v45, v47, v45, s[36:37]                  // 00000003A378: D100002D 00925B2F
	v_cndmask_b32_e64 v44, v46, v44, s[36:37]                  // 00000003A380: D100002C 0092592E
	v_lshlrev_b32_e32 v46, 16, v62                             // 00000003A388: 245C7C90
	v_and_b32_e32 v47, 0xffff0000, v62                         // 00000003A38C: 265E7CFF FFFF0000
	v_pk_add_f32 v[46:47], v[44:45], v[46:47]                  // 00000003A394: D3B2402E 18025D2C
	s_nop 0                                                    // 00000003A39C: BF800000
	v_cndmask_b32_e64 v46, v46, v44, s[40:41]                  // 00000003A3A0: D100002E 00A2592E
	v_and_b32_e32 v44, 0x7f800000, v46                         // 00000003A3A8: 26585CFF 7F800000
	v_cmp_ne_u32_e64 s[44:45], s58, v44                        // 00000003A3B0: D0CD002C 0002583A
	s_and_saveexec_b64 s[72:73], s[44:45]                      // 00000003A3B8: BEC8202C
	s_xor_b64 s[44:45], exec, s[72:73]                         // 00000003A3BC: 88AC487E
	v_bfe_u32 v44, v46, 16, 1                                  // 00000003A3C0: D1C8002C 0205212E
	v_add3_u32 v44, v46, v44, s59                              // 00000003A3C8: D1FF002C 00EE592E
	s_andn2_saveexec_b64 s[72:73], s[44:45]                    // 00000003A3D0: BEC8232C
	v_or_b32_e32 v44, 0x10000, v46                             // 00000003A3D4: 28585CFF 00010000
	v_cmp_eq_u32_sdwa s[44:45], v46, v21 src0_sel:WORD_0 src1_sel:DWORD// 00000003A3DC: 7D942AF9 0604AC2E
	s_nop 1                                                    // 00000003A3E4: BF800001
	v_cndmask_b32_e64 v44, v44, v46, s[44:45]                  // 00000003A3E8: D100002C 00B25D2C
	s_or_b64 exec, exec, s[72:73]                              // 00000003A3F0: 87FE487E
	v_cndmask_b32_e64 v45, v47, v45, s[40:41]                  // 00000003A3F4: D100002D 00A25B2F
	v_and_b32_e32 v46, 0x7f800000, v45                         // 00000003A3FC: 265C5AFF 7F800000
	v_cmp_ne_u32_e64 s[44:45], s58, v46                        // 00000003A404: D0CD002C 00025C3A
	s_and_saveexec_b64 s[48:49], s[44:45]                      // 00000003A40C: BEB0202C
	s_xor_b64 s[44:45], exec, s[48:49]                         // 00000003A410: 88AC307E
	v_bfe_u32 v46, v45, 16, 1                                  // 00000003A414: D1C8002E 0205212D
	v_add3_u32 v46, v45, v46, s59                              // 00000003A41C: D1FF002E 00EE5D2D
	s_andn2_saveexec_b64 s[72:73], s[44:45]                    // 00000003A424: BEC8232C
	s_cbranch_execz 65296                                      // 00000003A428: BF88FF10 <EpCombineIntraNodeKernel_bf16_nop2p+0x106c>
	v_or_b32_e32 v46, 0x10000, v45                             // 00000003A42C: 285C5AFF 00010000
	v_cmp_eq_u32_sdwa s[44:45], v45, v21 src0_sel:WORD_0 src1_sel:DWORD// 00000003A434: 7D942AF9 0604AC2D
	s_nop 1                                                    // 00000003A43C: BF800001
	v_cndmask_b32_e64 v46, v46, v45, s[44:45]                  // 00000003A440: D100002E 00B25B2E
	s_branch 65288                                             // 00000003A448: BF82FF08 <EpCombineIntraNodeKernel_bf16_nop2p+0x106c>
	s_or_b64 exec, exec, s[96:97]                              // 00000003A44C: 87FE607E
	v_readlane_b32 s48, v63, 13                                // 00000003A450: D2890030 00011B3F
	v_readlane_b32 s49, v63, 14                                // 00000003A458: D2890031 00011D3F
	s_or_b64 exec, exec, s[94:95]                              // 00000003A460: 87FE5E7E
	v_lshl_add_u64 v[38:39], v[44:45], 0, v[42:43]             // 00000003A464: D2080026 04A9012C
	v_cmp_lt_u64_e32 vcc, v[38:39], v[36:37]                   // 00000003A46C: 7DD24926
	s_and_saveexec_b64 s[24:25], vcc                           // 00000003A470: BE98206A
	s_cbranch_execz 187                                        // 00000003A474: BF8800BB <EpCombineIntraNodeKernel_bf16_nop2p+0x1764>
	ds_read2_b64 v[0:3], v27 offset1:1                         // 00000003A478: D8EE0100 0000001B
	ds_read2_b64 v[4:7], v27 offset0:2 offset1:3               // 00000003A480: D8EE0302 0400001B
	ds_read2_b64 v[8:11], v27 offset0:4 offset1:5              // 00000003A488: D8EE0504 0800001B
	ds_read2_b64 v[12:15], v27 offset0:6 offset1:7             // 00000003A490: D8EE0706 0C00001B
	ds_read2_b64 v[16:19], v27 offset0:8 offset1:9             // 00000003A498: D8EE0908 1000001B
	s_mov_b64 s[26:27], 0                                      // 00000003A4A0: BE9A0180
	s_waitcnt lgkmcnt(0)                                       // 00000003A4A4: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[0:1]                            // 00000003A4A8: 7DDA0080
	v_cmp_ne_u64_e64 s[4:5], 0, v[2:3]                         // 00000003A4AC: D0ED0004 00020480
	v_cmp_ne_u64_e64 s[6:7], 0, v[4:5]                         // 00000003A4B4: D0ED0006 00020880
	v_cmp_ne_u64_e64 s[8:9], 0, v[6:7]                         // 00000003A4BC: D0ED0008 00020C80
	v_cmp_ne_u64_e64 s[10:11], 0, v[8:9]                       // 00000003A4C4: D0ED000A 00021080
	v_cmp_ne_u64_e64 s[12:13], 0, v[10:11]                     // 00000003A4CC: D0ED000C 00021480
	v_cmp_ne_u64_e64 s[14:15], 0, v[12:13]                     // 00000003A4D4: D0ED000E 00021880
	v_cmp_ne_u64_e64 s[16:17], 0, v[14:15]                     // 00000003A4DC: D0ED0010 00021C80
	v_cmp_ne_u64_e64 s[18:19], 0, v[16:17]                     // 00000003A4E4: D0ED0012 00022080
	v_cmp_ne_u64_e64 s[20:21], 0, v[18:19]                     // 00000003A4EC: D0ED0014 00022480
	v_lshlrev_b64 v[40:41], 1, v[38:39]                        // 00000003A4F4: D28F0028 00024C81
	s_branch 32                                                // 00000003A4FC: BF820020 <EpCombineIntraNodeKernel_bf16_nop2p+0x1580>
	s_or_b64 exec, exec, s[28:29]                              // 00000003A500: 87FE1C7E
	v_lshl_add_u64 v[46:47], v[38:39], 1, v[34:35]             // 00000003A504: D208002E 04890326
	v_lshl_add_u64 v[38:39], v[38:39], 0, 64                   // 00000003A50C: D2080026 03010126
	v_cmp_ge_u64_e64 s[22:23], v[38:39], v[36:37]              // 00000003A514: D0EE0016 00024926
	v_lshl_add_u64 v[18:19], v[18:19], 0, s[86:87]             // 00000003A51C: D2080012 01590112
	v_lshl_add_u64 v[16:17], v[16:17], 0, s[86:87]             // 00000003A524: D2080010 01590110
	v_lshl_add_u64 v[14:15], v[14:15], 0, s[86:87]             // 00000003A52C: D208000E 0159010E
	v_lshl_add_u64 v[12:13], v[12:13], 0, s[86:87]             // 00000003A534: D208000C 0159010C
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[86:87]             // 00000003A53C: D208000A 0159010A
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[86:87]                 // 00000003A544: D2080008 01590108
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[86:87]                 // 00000003A54C: D2080006 01590106
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[86:87]                 // 00000003A554: D2080004 01590104
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[86:87]                 // 00000003A55C: D2080002 01590102
	s_or_b64 s[26:27], s[22:23], s[26:27]                      // 00000003A564: 879A1A16
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[86:87]                 // 00000003A568: D2080000 01590100
	flat_store_short_d16_hi v[46:47], v44                      // 00000003A570: DC6C0000 00002C2E
	s_andn2_b64 exec, exec, s[26:27]                           // 00000003A578: 89FE1A7E
	s_cbranch_execz 121                                        // 00000003A57C: BF880079 <EpCombineIntraNodeKernel_bf16_nop2p+0x1764>
	v_mov_b32_e32 v20, 0                                       // 00000003A580: 7E280280
	s_and_saveexec_b64 s[22:23], vcc                           // 00000003A584: BE96206A
	s_cbranch_execz 7                                          // 00000003A588: BF880007 <EpCombineIntraNodeKernel_bf16_nop2p+0x15a8>
	v_lshl_add_u64 v[44:45], v[0:1], 0, v[40:41]               // 00000003A58C: D208002C 04A10100
	flat_load_ushort v20, v[44:45]                             // 00000003A594: DC480000 1400002C
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003A59C: BF8C0070
	v_lshlrev_b32_e32 v20, 16, v20                             // 00000003A5A0: 24282890
	v_add_f32_e32 v20, 0, v20                                  // 00000003A5A4: 02282880
	s_or_b64 exec, exec, s[22:23]                              // 00000003A5A8: 87FE167E
	s_and_saveexec_b64 s[22:23], s[4:5]                        // 00000003A5AC: BE962004
	s_cbranch_execz 7                                          // 00000003A5B0: BF880007 <EpCombineIntraNodeKernel_bf16_nop2p+0x15d0>
	v_lshl_add_u64 v[44:45], v[2:3], 0, v[40:41]               // 00000003A5B4: D208002C 04A10102
	flat_load_ushort v44, v[44:45]                             // 00000003A5BC: DC480000 2C00002C
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003A5C4: BF8C0070
	v_lshlrev_b32_e32 v44, 16, v44                             // 00000003A5C8: 24585890
	v_add_f32_e32 v20, v20, v44                                // 00000003A5CC: 02285914
	s_or_b64 exec, exec, s[22:23]                              // 00000003A5D0: 87FE167E
	s_and_saveexec_b64 s[22:23], s[6:7]                        // 00000003A5D4: BE962006
	s_cbranch_execz 7                                          // 00000003A5D8: BF880007 <EpCombineIntraNodeKernel_bf16_nop2p+0x15f8>
	v_lshl_add_u64 v[44:45], v[4:5], 0, v[40:41]               // 00000003A5DC: D208002C 04A10104
	flat_load_ushort v44, v[44:45]                             // 00000003A5E4: DC480000 2C00002C
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003A5EC: BF8C0070
	v_lshlrev_b32_e32 v44, 16, v44                             // 00000003A5F0: 24585890
	v_add_f32_e32 v20, v20, v44                                // 00000003A5F4: 02285914
	s_or_b64 exec, exec, s[22:23]                              // 00000003A5F8: 87FE167E
	s_and_saveexec_b64 s[22:23], s[8:9]                        // 00000003A5FC: BE962008
	s_cbranch_execz 7                                          // 00000003A600: BF880007 <EpCombineIntraNodeKernel_bf16_nop2p+0x1620>
	v_lshl_add_u64 v[44:45], v[6:7], 0, v[40:41]               // 00000003A604: D208002C 04A10106
	flat_load_ushort v44, v[44:45]                             // 00000003A60C: DC480000 2C00002C
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003A614: BF8C0070
	v_lshlrev_b32_e32 v44, 16, v44                             // 00000003A618: 24585890
	v_add_f32_e32 v20, v20, v44                                // 00000003A61C: 02285914
	s_or_b64 exec, exec, s[22:23]                              // 00000003A620: 87FE167E
	s_and_saveexec_b64 s[22:23], s[10:11]                      // 00000003A624: BE96200A
	s_cbranch_execz 7                                          // 00000003A628: BF880007 <EpCombineIntraNodeKernel_bf16_nop2p+0x1648>
	v_lshl_add_u64 v[44:45], v[8:9], 0, v[40:41]               // 00000003A62C: D208002C 04A10108
	flat_load_ushort v44, v[44:45]                             // 00000003A634: DC480000 2C00002C
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003A63C: BF8C0070
	v_lshlrev_b32_e32 v44, 16, v44                             // 00000003A640: 24585890
	v_add_f32_e32 v20, v20, v44                                // 00000003A644: 02285914
	s_or_b64 exec, exec, s[22:23]                              // 00000003A648: 87FE167E
	s_and_saveexec_b64 s[22:23], s[12:13]                      // 00000003A64C: BE96200C
	s_cbranch_execz 7                                          // 00000003A650: BF880007 <EpCombineIntraNodeKernel_bf16_nop2p+0x1670>
	v_lshl_add_u64 v[44:45], v[10:11], 0, v[40:41]             // 00000003A654: D208002C 04A1010A
	flat_load_ushort v44, v[44:45]                             // 00000003A65C: DC480000 2C00002C
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003A664: BF8C0070
	v_lshlrev_b32_e32 v44, 16, v44                             // 00000003A668: 24585890
	v_add_f32_e32 v20, v20, v44                                // 00000003A66C: 02285914
	s_or_b64 exec, exec, s[22:23]                              // 00000003A670: 87FE167E
	s_and_saveexec_b64 s[22:23], s[14:15]                      // 00000003A674: BE96200E
	s_cbranch_execz 7                                          // 00000003A678: BF880007 <EpCombineIntraNodeKernel_bf16_nop2p+0x1698>
	v_lshl_add_u64 v[44:45], v[12:13], 0, v[40:41]             // 00000003A67C: D208002C 04A1010C
	flat_load_ushort v44, v[44:45]                             // 00000003A684: DC480000 2C00002C
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003A68C: BF8C0070
	v_lshlrev_b32_e32 v44, 16, v44                             // 00000003A690: 24585890
	v_add_f32_e32 v20, v20, v44                                // 00000003A694: 02285914
	s_or_b64 exec, exec, s[22:23]                              // 00000003A698: 87FE167E
	s_and_saveexec_b64 s[22:23], s[16:17]                      // 00000003A69C: BE962010
	s_cbranch_execz 7                                          // 00000003A6A0: BF880007 <EpCombineIntraNodeKernel_bf16_nop2p+0x16c0>
	v_lshl_add_u64 v[44:45], v[14:15], 0, v[40:41]             // 00000003A6A4: D208002C 04A1010E
	flat_load_ushort v44, v[44:45]                             // 00000003A6AC: DC480000 2C00002C
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003A6B4: BF8C0070
	v_lshlrev_b32_e32 v44, 16, v44                             // 00000003A6B8: 24585890
	v_add_f32_e32 v20, v20, v44                                // 00000003A6BC: 02285914
	s_or_b64 exec, exec, s[22:23]                              // 00000003A6C0: 87FE167E
	s_and_saveexec_b64 s[22:23], s[18:19]                      // 00000003A6C4: BE962012
	s_cbranch_execz 7                                          // 00000003A6C8: BF880007 <EpCombineIntraNodeKernel_bf16_nop2p+0x16e8>
	v_lshl_add_u64 v[44:45], v[16:17], 0, v[40:41]             // 00000003A6CC: D208002C 04A10110
	flat_load_ushort v44, v[44:45]                             // 00000003A6D4: DC480000 2C00002C
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003A6DC: BF8C0070
	v_lshlrev_b32_e32 v44, 16, v44                             // 00000003A6E0: 24585890
	v_add_f32_e32 v20, v20, v44                                // 00000003A6E4: 02285914
	s_or_b64 exec, exec, s[22:23]                              // 00000003A6E8: 87FE167E
	s_and_saveexec_b64 s[22:23], s[20:21]                      // 00000003A6EC: BE962014
	s_cbranch_execz 7                                          // 00000003A6F0: BF880007 <EpCombineIntraNodeKernel_bf16_nop2p+0x1710>
	v_lshl_add_u64 v[44:45], v[18:19], 0, v[40:41]             // 00000003A6F4: D208002C 04A10112
	flat_load_ushort v44, v[44:45]                             // 00000003A6FC: DC480000 2C00002C
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003A704: BF8C0070
	v_lshlrev_b32_e32 v44, 16, v44                             // 00000003A708: 24585890
	v_add_f32_e32 v20, v20, v44                                // 00000003A70C: 02285914
	s_or_b64 exec, exec, s[22:23]                              // 00000003A710: 87FE167E
	v_and_b32_e32 v44, 0x7f800000, v20                         // 00000003A714: 265828FF 7F800000
	v_cmp_ne_u32_e64 s[22:23], s58, v44                        // 00000003A71C: D0CD0016 0002583A
	s_and_saveexec_b64 s[28:29], s[22:23]                      // 00000003A724: BE9C2016
	s_xor_b64 s[22:23], exec, s[28:29]                         // 00000003A728: 88961C7E
	v_bfe_u32 v44, v20, 16, 1                                  // 00000003A72C: D1C8002C 02052114
	v_add3_u32 v44, v20, v44, s59                              // 00000003A734: D1FF002C 00EE5914
	s_andn2_saveexec_b64 s[28:29], s[22:23]                    // 00000003A73C: BE9C2316
	s_cbranch_execz 65391                                      // 00000003A740: BF88FF6F <EpCombineIntraNodeKernel_bf16_nop2p+0x1500>
	v_or_b32_e32 v44, 0x10000, v20                             // 00000003A744: 285828FF 00010000
	v_cmp_eq_u32_sdwa s[22:23], v20, v21 src0_sel:WORD_0 src1_sel:DWORD// 00000003A74C: 7D942AF9 06049614
	s_nop 1                                                    // 00000003A754: BF800001
	v_cndmask_b32_e64 v44, v44, v20, s[22:23]                  // 00000003A758: D100002C 005A292C
	s_branch 65383                                             // 00000003A760: BF82FF67 <EpCombineIntraNodeKernel_bf16_nop2p+0x1500>
	s_or_b64 exec, exec, s[24:25]                              // 00000003A764: 87FE187E
	s_mov_b64 s[4:5], 0                                        // 00000003A768: BE840180
	s_mov_b64 s[6:7], 0                                        // 00000003A76C: BE860180
	s_and_b64 vcc, exec, s[6:7]                                // 00000003A770: 86EA067E
	s_cbranch_vccz 431                                         // 00000003A774: BF8601AF <EpCombineIntraNodeKernel_bf16_nop2p+0x1e34>
	s_cmp_eq_u32 s90, 8                                        // 00000003A778: BF06885A
	s_mov_b64 s[4:5], -1                                       // 00000003A77C: BE8401C1
	s_cbranch_scc0 428                                         // 00000003A780: BF8401AC <EpCombineIntraNodeKernel_bf16_nop2p+0x1e34>
	v_cmp_lt_u64_e32 vcc, s[84:85], v[36:37]                   // 00000003A784: 7DD24854
	v_mov_b64_e32 v[38:39], 0                                  // 00000003A788: 7E4C7080
	s_and_saveexec_b64 s[38:39], vcc                           // 00000003A78C: BEA6206A
	s_cbranch_execz 259                                        // 00000003A790: BF880103 <EpCombineIntraNodeKernel_bf16_nop2p+0x1ba0>
	ds_read2_b64 v[0:3], v27 offset0:2 offset1:3               // 00000003A794: D8EE0302 0000001B
	ds_read2_b64 v[4:7], v27 offset1:1                         // 00000003A79C: D8EE0100 0400001B
	ds_read2_b64 v[8:11], v27 offset0:6 offset1:7              // 00000003A7A4: D8EE0706 0800001B
	ds_read2_b64 v[12:15], v27 offset0:4 offset1:5             // 00000003A7AC: D8EE0504 0C00001B
	v_lshlrev_b32_e32 v20, 1, v26                              // 00000003A7B4: 24283481
	v_lshl_add_u64 v[16:17], v[34:35], 0, v[20:21]             // 00000003A7B8: D2080010 04510122
	v_lshrrev_b64 v[18:19], 7, v[36:37]                        // 00000003A7C0: D2900012 00024887
	s_mov_b64 s[40:41], 0                                      // 00000003A7C8: BEA80180
	s_waitcnt lgkmcnt(0)                                       // 00000003A7CC: BF8CC07F
	v_cmp_eq_u64_e32 vcc, 0, v[4:5]                            // 00000003A7D0: 7DD40880
	v_cmp_ne_u64_e64 s[4:5], 0, v[4:5]                         // 00000003A7D4: D0ED0004 00020880
	v_cmp_eq_u64_e64 s[6:7], 0, v[6:7]                         // 00000003A7DC: D0EA0006 00020C80
	v_cmp_ne_u64_e64 s[8:9], 0, v[6:7]                         // 00000003A7E4: D0ED0008 00020C80
	v_cmp_eq_u64_e64 s[10:11], 0, v[0:1]                       // 00000003A7EC: D0EA000A 00020080
	v_cmp_ne_u64_e64 s[12:13], 0, v[0:1]                       // 00000003A7F4: D0ED000C 00020080
	v_cmp_eq_u64_e64 s[14:15], 0, v[2:3]                       // 00000003A7FC: D0EA000E 00020480
	v_cmp_ne_u64_e64 s[16:17], 0, v[2:3]                       // 00000003A804: D0ED0010 00020480
	v_cmp_eq_u64_e64 s[18:19], 0, v[12:13]                     // 00000003A80C: D0EA0012 00021880
	v_cmp_ne_u64_e64 s[20:21], 0, v[12:13]                     // 00000003A814: D0ED0014 00021880
	v_cmp_eq_u64_e64 s[22:23], 0, v[14:15]                     // 00000003A81C: D0EA0016 00021C80
	v_cmp_ne_u64_e64 s[24:25], 0, v[14:15]                     // 00000003A824: D0ED0018 00021C80
	v_cmp_eq_u64_e64 s[26:27], 0, v[8:9]                       // 00000003A82C: D0EA001A 00021080
	v_cmp_ne_u64_e64 s[28:29], 0, v[8:9]                       // 00000003A834: D0ED001C 00021080
	v_cmp_eq_u64_e64 s[30:31], 0, v[10:11]                     // 00000003A83C: D0EA001E 00021480
	v_cmp_ne_u64_e64 s[34:35], 0, v[10:11]                     // 00000003A844: D0ED0022 00021480
	s_mov_b64 s[42:43], 0                                      // 00000003A84C: BEAA0180
	s_branch 35                                                // 00000003A850: BF820023 <EpCombineIntraNodeKernel_bf16_nop2p+0x18e0>
	s_or_b64 exec, exec, s[44:45]                              // 00000003A854: 87FE2C7E
	v_lshrrev_b32_e32 v41, 16, v38                             // 00000003A858: 20524C90
	v_lshl_add_u64 v[38:39], s[42:43], 1, v[16:17]             // 00000003A85C: D2080026 0441022A
	s_add_u32 s42, s42, 0x80                                   // 00000003A864: 802AFF2A 00000080
	v_lshl_add_u64 v[18:19], v[18:19], 0, -1                   // 00000003A86C: D2080012 03050112
	v_and_or_b32 v40, v40, s82, v41                            // 00000003A874: D2010028 04A4A528
	s_addc_u32 s43, s43, 0                                     // 00000003A87C: 822B802B
	v_cmp_eq_u64_e64 s[36:37], 0, v[18:19]                     // 00000003A880: D0EA0024 00022480
	flat_store_dword v[38:39], v40 nt                          // 00000003A888: DC720000 00002826
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[88:89]             // 00000003A890: D208000A 0161010A
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[88:89]                 // 00000003A898: D2080008 01610108
	v_lshl_add_u64 v[14:15], v[14:15], 0, s[88:89]             // 00000003A8A0: D208000E 0161010E
	v_lshl_add_u64 v[12:13], v[12:13], 0, s[88:89]             // 00000003A8A8: D208000C 0161010C
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[88:89]                 // 00000003A8B0: D2080002 01610102
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[88:89]                 // 00000003A8B8: D2080000 01610100
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[88:89]                 // 00000003A8C0: D2080006 01610106
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[88:89]                 // 00000003A8C8: D2080004 01610104
	s_or_b64 s[40:41], s[36:37], s[40:41]                      // 00000003A8D0: 87A82824
	v_mov_b64_e32 v[38:39], s[42:43]                           // 00000003A8D4: 7E4C702A
	s_andn2_b64 exec, exec, s[40:41]                           // 00000003A8D8: 89FE287E
	s_cbranch_execz 175                                        // 00000003A8DC: BF8800AF <EpCombineIntraNodeKernel_bf16_nop2p+0x1b9c>
	s_and_saveexec_b64 s[36:37], s[4:5]                        // 00000003A8E0: BEA42004
	s_cbranch_execz 4                                          // 00000003A8E4: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x18f8>
	v_lshl_add_u64 v[38:39], v[4:5], 0, v[30:31]               // 00000003A8E8: D2080026 04790104
	flat_load_dword v54, v[38:39] nt                           // 00000003A8F0: DC520000 36000026
	s_or_b64 exec, exec, s[36:37]                              // 00000003A8F8: 87FE247E
	s_and_saveexec_b64 s[36:37], s[8:9]                        // 00000003A8FC: BEA42008
	s_cbranch_execz 4                                          // 00000003A900: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x1914>
	v_lshl_add_u64 v[38:39], v[6:7], 0, v[30:31]               // 00000003A904: D2080026 04790106
	flat_load_dword v47, v[38:39] nt                           // 00000003A90C: DC520000 2F000026
	s_or_b64 exec, exec, s[36:37]                              // 00000003A914: 87FE247E
	s_and_saveexec_b64 s[36:37], s[12:13]                      // 00000003A918: BEA4200C
	s_cbranch_execz 4                                          // 00000003A91C: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x1930>
	v_lshl_add_u64 v[38:39], v[0:1], 0, v[30:31]               // 00000003A920: D2080026 04790100
	flat_load_dword v46, v[38:39] nt                           // 00000003A928: DC520000 2E000026
	s_or_b64 exec, exec, s[36:37]                              // 00000003A930: 87FE247E
	s_and_saveexec_b64 s[36:37], s[16:17]                      // 00000003A934: BEA42010
	s_cbranch_execz 4                                          // 00000003A938: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x194c>
	v_lshl_add_u64 v[38:39], v[2:3], 0, v[30:31]               // 00000003A93C: D2080026 04790102
	flat_load_dword v45, v[38:39] nt                           // 00000003A944: DC520000 2D000026
	s_or_b64 exec, exec, s[36:37]                              // 00000003A94C: 87FE247E
	s_and_saveexec_b64 s[36:37], s[20:21]                      // 00000003A950: BEA42014
	s_cbranch_execz 4                                          // 00000003A954: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x1968>
	v_lshl_add_u64 v[38:39], v[12:13], 0, v[30:31]             // 00000003A958: D2080026 0479010C
	flat_load_dword v44, v[38:39] nt                           // 00000003A960: DC520000 2C000026
	s_or_b64 exec, exec, s[36:37]                              // 00000003A968: 87FE247E
	s_and_saveexec_b64 s[36:37], s[24:25]                      // 00000003A96C: BEA42018
	s_cbranch_execz 4                                          // 00000003A970: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x1984>
	v_lshl_add_u64 v[38:39], v[14:15], 0, v[30:31]             // 00000003A974: D2080026 0479010E
	flat_load_dword v20, v[38:39] nt                           // 00000003A97C: DC520000 14000026
	s_or_b64 exec, exec, s[36:37]                              // 00000003A984: 87FE247E
	s_and_saveexec_b64 s[36:37], s[28:29]                      // 00000003A988: BEA4201C
	s_cbranch_execz 4                                          // 00000003A98C: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x19a0>
	v_lshl_add_u64 v[38:39], v[8:9], 0, v[30:31]               // 00000003A990: D2080026 04790108
	flat_load_dword v55, v[38:39] nt                           // 00000003A998: DC520000 37000026
	s_or_b64 exec, exec, s[36:37]                              // 00000003A9A0: 87FE247E
	s_and_saveexec_b64 s[36:37], s[34:35]                      // 00000003A9A4: BEA42022
	s_cbranch_execz 4                                          // 00000003A9A8: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x19bc>
	v_lshl_add_u64 v[38:39], v[10:11], 0, v[30:31]             // 00000003A9AC: D2080026 0479010A
	flat_load_dword v56, v[38:39] nt                           // 00000003A9B4: DC520000 38000026
	s_or_b64 exec, exec, s[36:37]                              // 00000003A9BC: 87FE247E
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003A9C0: BF8C0070
	v_lshlrev_b32_e32 v38, 16, v54                             // 00000003A9C4: 244C6C90
	v_and_b32_e32 v39, 0xffff0000, v54                         // 00000003A9C8: 264E6CFF FFFF0000
	v_pk_add_f32 v[38:39], v[38:39], 0 op_sel_hi:[1,0]         // 00000003A9D0: D3B24026 08010126
	v_lshlrev_b32_e32 v40, 16, v47                             // 00000003A9D8: 24505E90
	v_cndmask_b32_e64 v39, v39, 0, vcc                         // 00000003A9DC: D1000027 01A90127
	v_cndmask_b32_e64 v38, v38, 0, vcc                         // 00000003A9E4: D1000026 01A90126
	v_and_b32_e32 v41, 0xffff0000, v47                         // 00000003A9EC: 26525EFF FFFF0000
	v_pk_add_f32 v[40:41], v[38:39], v[40:41]                  // 00000003A9F4: D3B24028 18025126
	s_nop 0                                                    // 00000003A9FC: BF800000
	v_cndmask_b32_e64 v39, v41, v39, s[6:7]                    // 00000003AA00: D1000027 001A4F29
	v_cndmask_b32_e64 v38, v40, v38, s[6:7]                    // 00000003AA08: D1000026 001A4D28
	v_lshlrev_b32_e32 v40, 16, v46                             // 00000003AA10: 24505C90
	v_and_b32_e32 v41, 0xffff0000, v46                         // 00000003AA14: 26525CFF FFFF0000
	v_pk_add_f32 v[40:41], v[38:39], v[40:41]                  // 00000003AA1C: D3B24028 18025126
	s_nop 0                                                    // 00000003AA24: BF800000
	v_cndmask_b32_e64 v39, v41, v39, s[10:11]                  // 00000003AA28: D1000027 002A4F29
	v_cndmask_b32_e64 v38, v40, v38, s[10:11]                  // 00000003AA30: D1000026 002A4D28
	v_lshlrev_b32_e32 v40, 16, v45                             // 00000003AA38: 24505A90
	v_and_b32_e32 v41, 0xffff0000, v45                         // 00000003AA3C: 26525AFF FFFF0000
	v_pk_add_f32 v[40:41], v[38:39], v[40:41]                  // 00000003AA44: D3B24028 18025126
	s_nop 0                                                    // 00000003AA4C: BF800000
	v_cndmask_b32_e64 v39, v41, v39, s[14:15]                  // 00000003AA50: D1000027 003A4F29
	v_cndmask_b32_e64 v38, v40, v38, s[14:15]                  // 00000003AA58: D1000026 003A4D28
	v_lshlrev_b32_e32 v40, 16, v44                             // 00000003AA60: 24505890
	v_and_b32_e32 v41, 0xffff0000, v44                         // 00000003AA64: 265258FF FFFF0000
	v_pk_add_f32 v[40:41], v[38:39], v[40:41]                  // 00000003AA6C: D3B24028 18025126
	s_nop 0                                                    // 00000003AA74: BF800000
	v_cndmask_b32_e64 v39, v41, v39, s[18:19]                  // 00000003AA78: D1000027 004A4F29
	v_cndmask_b32_e64 v38, v40, v38, s[18:19]                  // 00000003AA80: D1000026 004A4D28
	v_lshlrev_b32_e32 v40, 16, v20                             // 00000003AA88: 24502890
	v_and_b32_e32 v41, 0xffff0000, v20                         // 00000003AA8C: 265228FF FFFF0000
	v_pk_add_f32 v[40:41], v[38:39], v[40:41]                  // 00000003AA94: D3B24028 18025126
	s_nop 0                                                    // 00000003AA9C: BF800000
	v_cndmask_b32_e64 v39, v41, v39, s[22:23]                  // 00000003AAA0: D1000027 005A4F29
	v_cndmask_b32_e64 v38, v40, v38, s[22:23]                  // 00000003AAA8: D1000026 005A4D28
	v_lshlrev_b32_e32 v40, 16, v55                             // 00000003AAB0: 24506E90
	v_and_b32_e32 v41, 0xffff0000, v55                         // 00000003AAB4: 26526EFF FFFF0000
	v_pk_add_f32 v[40:41], v[38:39], v[40:41]                  // 00000003AABC: D3B24028 18025126
	s_nop 0                                                    // 00000003AAC4: BF800000
	v_cndmask_b32_e64 v39, v41, v39, s[26:27]                  // 00000003AAC8: D1000027 006A4F29
	v_cndmask_b32_e64 v38, v40, v38, s[26:27]                  // 00000003AAD0: D1000026 006A4D28
	v_lshlrev_b32_e32 v40, 16, v56                             // 00000003AAD8: 24507090
	v_and_b32_e32 v41, 0xffff0000, v56                         // 00000003AADC: 265270FF FFFF0000
	v_pk_add_f32 v[40:41], v[38:39], v[40:41]                  // 00000003AAE4: D3B24028 18025126
	s_nop 0                                                    // 00000003AAEC: BF800000
	v_cndmask_b32_e64 v40, v40, v38, s[30:31]                  // 00000003AAF0: D1000028 007A4D28
	v_and_b32_e32 v38, 0x7f800000, v40                         // 00000003AAF8: 264C50FF 7F800000
	v_cmp_ne_u32_e64 s[36:37], s58, v38                        // 00000003AB00: D0CD0024 00024C3A
	s_and_saveexec_b64 s[44:45], s[36:37]                      // 00000003AB08: BEAC2024
	s_xor_b64 s[36:37], exec, s[44:45]                         // 00000003AB0C: 88A42C7E
	v_bfe_u32 v38, v40, 16, 1                                  // 00000003AB10: D1C80026 02052128
	v_add3_u32 v38, v40, v38, s59                              // 00000003AB18: D1FF0026 00EE4D28
	s_andn2_saveexec_b64 s[44:45], s[36:37]                    // 00000003AB20: BEAC2324
	v_or_b32_e32 v38, 0x10000, v40                             // 00000003AB24: 284C50FF 00010000
	v_cmp_eq_u32_sdwa s[36:37], v40, v21 src0_sel:WORD_0 src1_sel:DWORD// 00000003AB2C: 7D942AF9 0604A428
	s_nop 1                                                    // 00000003AB34: BF800001
	v_cndmask_b32_e64 v38, v38, v40, s[36:37]                  // 00000003AB38: D1000026 00925126
	s_or_b64 exec, exec, s[44:45]                              // 00000003AB40: 87FE2C7E
	v_cndmask_b32_e64 v39, v41, v39, s[30:31]                  // 00000003AB44: D1000027 007A4F29
	v_and_b32_e32 v40, 0x7f800000, v39                         // 00000003AB4C: 26504EFF 7F800000
	v_cmp_ne_u32_e64 s[36:37], s58, v40                        // 00000003AB54: D0CD0024 0002503A
	s_and_saveexec_b64 s[44:45], s[36:37]                      // 00000003AB5C: BEAC2024
	s_xor_b64 s[36:37], exec, s[44:45]                         // 00000003AB60: 88A42C7E
	v_bfe_u32 v40, v39, 16, 1                                  // 00000003AB64: D1C80028 02052127
	v_add3_u32 v40, v39, v40, s59                              // 00000003AB6C: D1FF0028 00EE5127
	s_andn2_saveexec_b64 s[44:45], s[36:37]                    // 00000003AB74: BEAC2324
	s_cbranch_execz 65334                                      // 00000003AB78: BF88FF36 <EpCombineIntraNodeKernel_bf16_nop2p+0x1854>
	v_or_b32_e32 v40, 0x10000, v39                             // 00000003AB7C: 28504EFF 00010000
	v_cmp_eq_u32_sdwa s[36:37], v39, v21 src0_sel:WORD_0 src1_sel:DWORD// 00000003AB84: 7D942AF9 0604A427
	s_nop 1                                                    // 00000003AB8C: BF800001
	v_cndmask_b32_e64 v40, v40, v39, s[36:37]                  // 00000003AB90: D1000028 00924F28
	s_branch 65326                                             // 00000003AB98: BF82FF2E <EpCombineIntraNodeKernel_bf16_nop2p+0x1854>
	s_or_b64 exec, exec, s[40:41]                              // 00000003AB9C: 87FE287E
	s_or_b64 exec, exec, s[38:39]                              // 00000003ABA0: 87FE267E
	v_lshl_add_u64 v[16:17], v[38:39], 0, v[42:43]             // 00000003ABA4: D2080010 04A90126
	v_cmp_lt_u64_e32 vcc, v[16:17], v[36:37]                   // 00000003ABAC: 7DD24910
	s_and_saveexec_b64 s[20:21], vcc                           // 00000003ABB0: BE94206A
	s_cbranch_execz 157                                        // 00000003ABB4: BF88009D <EpCombineIntraNodeKernel_bf16_nop2p+0x1e2c>
	ds_read2_b64 v[0:3], v27 offset1:1                         // 00000003ABB8: D8EE0100 0000001B
	ds_read2_b64 v[4:7], v27 offset0:2 offset1:3               // 00000003ABC0: D8EE0302 0400001B
	ds_read2_b64 v[8:11], v27 offset0:4 offset1:5              // 00000003ABC8: D8EE0504 0800001B
	ds_read2_b64 v[12:15], v27 offset0:6 offset1:7             // 00000003ABD0: D8EE0706 0C00001B
	s_mov_b64 s[22:23], 0                                      // 00000003ABD8: BE960180
	v_lshlrev_b64 v[18:19], 1, v[16:17]                        // 00000003ABDC: D28F0012 00022081
	s_waitcnt lgkmcnt(0)                                       // 00000003ABE4: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[0:1]                            // 00000003ABE8: 7DDA0080
	v_cmp_ne_u64_e64 s[4:5], 0, v[2:3]                         // 00000003ABEC: D0ED0004 00020480
	v_cmp_ne_u64_e64 s[6:7], 0, v[4:5]                         // 00000003ABF4: D0ED0006 00020880
	v_cmp_ne_u64_e64 s[8:9], 0, v[6:7]                         // 00000003ABFC: D0ED0008 00020C80
	v_cmp_ne_u64_e64 s[10:11], 0, v[8:9]                       // 00000003AC04: D0ED000A 00021080
	v_cmp_ne_u64_e64 s[12:13], 0, v[10:11]                     // 00000003AC0C: D0ED000C 00021480
	v_cmp_ne_u64_e64 s[14:15], 0, v[12:13]                     // 00000003AC14: D0ED000E 00021880
	v_cmp_ne_u64_e64 s[16:17], 0, v[14:15]                     // 00000003AC1C: D0ED0010 00021C80
	s_branch 28                                                // 00000003AC24: BF82001C <EpCombineIntraNodeKernel_bf16_nop2p+0x1c98>
	s_or_b64 exec, exec, s[24:25]                              // 00000003AC28: 87FE187E
	v_lshl_add_u64 v[40:41], v[16:17], 1, v[34:35]             // 00000003AC2C: D2080028 04890310
	v_lshl_add_u64 v[16:17], v[16:17], 0, 64                   // 00000003AC34: D2080010 03010110
	v_cmp_ge_u64_e64 s[18:19], v[16:17], v[36:37]              // 00000003AC3C: D0EE0012 00024910
	v_lshl_add_u64 v[14:15], v[14:15], 0, s[86:87]             // 00000003AC44: D208000E 0159010E
	v_lshl_add_u64 v[12:13], v[12:13], 0, s[86:87]             // 00000003AC4C: D208000C 0159010C
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[86:87]             // 00000003AC54: D208000A 0159010A
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[86:87]                 // 00000003AC5C: D2080008 01590108
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[86:87]                 // 00000003AC64: D2080006 01590106
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[86:87]                 // 00000003AC6C: D2080004 01590104
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[86:87]                 // 00000003AC74: D2080002 01590102
	s_or_b64 s[22:23], s[18:19], s[22:23]                      // 00000003AC7C: 87961612
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[86:87]                 // 00000003AC80: D2080000 01590100
	flat_store_short_d16_hi v[40:41], v38                      // 00000003AC88: DC6C0000 00002628
	s_andn2_b64 exec, exec, s[22:23]                           // 00000003AC90: 89FE167E
	s_cbranch_execz 101                                        // 00000003AC94: BF880065 <EpCombineIntraNodeKernel_bf16_nop2p+0x1e2c>
	v_mov_b32_e32 v20, 0                                       // 00000003AC98: 7E280280
	s_and_saveexec_b64 s[18:19], vcc                           // 00000003AC9C: BE92206A
	s_cbranch_execz 7                                          // 00000003ACA0: BF880007 <EpCombineIntraNodeKernel_bf16_nop2p+0x1cc0>
	v_lshl_add_u64 v[38:39], v[0:1], 0, v[18:19]               // 00000003ACA4: D2080026 04490100
	flat_load_ushort v20, v[38:39]                             // 00000003ACAC: DC480000 14000026
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003ACB4: BF8C0070
	v_lshlrev_b32_e32 v20, 16, v20                             // 00000003ACB8: 24282890
	v_add_f32_e32 v20, 0, v20                                  // 00000003ACBC: 02282880
	s_or_b64 exec, exec, s[18:19]                              // 00000003ACC0: 87FE127E
	s_and_saveexec_b64 s[18:19], s[4:5]                        // 00000003ACC4: BE922004
	s_cbranch_execz 7                                          // 00000003ACC8: BF880007 <EpCombineIntraNodeKernel_bf16_nop2p+0x1ce8>
	v_lshl_add_u64 v[38:39], v[2:3], 0, v[18:19]               // 00000003ACCC: D2080026 04490102
	flat_load_ushort v38, v[38:39]                             // 00000003ACD4: DC480000 26000026
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003ACDC: BF8C0070
	v_lshlrev_b32_e32 v38, 16, v38                             // 00000003ACE0: 244C4C90
	v_add_f32_e32 v20, v20, v38                                // 00000003ACE4: 02284D14
	s_or_b64 exec, exec, s[18:19]                              // 00000003ACE8: 87FE127E
	s_and_saveexec_b64 s[18:19], s[6:7]                        // 00000003ACEC: BE922006
	s_cbranch_execz 7                                          // 00000003ACF0: BF880007 <EpCombineIntraNodeKernel_bf16_nop2p+0x1d10>
	v_lshl_add_u64 v[38:39], v[4:5], 0, v[18:19]               // 00000003ACF4: D2080026 04490104
	flat_load_ushort v38, v[38:39]                             // 00000003ACFC: DC480000 26000026
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003AD04: BF8C0070
	v_lshlrev_b32_e32 v38, 16, v38                             // 00000003AD08: 244C4C90
	v_add_f32_e32 v20, v20, v38                                // 00000003AD0C: 02284D14
	s_or_b64 exec, exec, s[18:19]                              // 00000003AD10: 87FE127E
	s_and_saveexec_b64 s[18:19], s[8:9]                        // 00000003AD14: BE922008
	s_cbranch_execz 7                                          // 00000003AD18: BF880007 <EpCombineIntraNodeKernel_bf16_nop2p+0x1d38>
	v_lshl_add_u64 v[38:39], v[6:7], 0, v[18:19]               // 00000003AD1C: D2080026 04490106
	flat_load_ushort v38, v[38:39]                             // 00000003AD24: DC480000 26000026
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003AD2C: BF8C0070
	v_lshlrev_b32_e32 v38, 16, v38                             // 00000003AD30: 244C4C90
	v_add_f32_e32 v20, v20, v38                                // 00000003AD34: 02284D14
	s_or_b64 exec, exec, s[18:19]                              // 00000003AD38: 87FE127E
	s_and_saveexec_b64 s[18:19], s[10:11]                      // 00000003AD3C: BE92200A
	s_cbranch_execz 7                                          // 00000003AD40: BF880007 <EpCombineIntraNodeKernel_bf16_nop2p+0x1d60>
	v_lshl_add_u64 v[38:39], v[8:9], 0, v[18:19]               // 00000003AD44: D2080026 04490108
	flat_load_ushort v38, v[38:39]                             // 00000003AD4C: DC480000 26000026
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003AD54: BF8C0070
	v_lshlrev_b32_e32 v38, 16, v38                             // 00000003AD58: 244C4C90
	v_add_f32_e32 v20, v20, v38                                // 00000003AD5C: 02284D14
	s_or_b64 exec, exec, s[18:19]                              // 00000003AD60: 87FE127E
	s_and_saveexec_b64 s[18:19], s[12:13]                      // 00000003AD64: BE92200C
	s_cbranch_execz 7                                          // 00000003AD68: BF880007 <EpCombineIntraNodeKernel_bf16_nop2p+0x1d88>
	v_lshl_add_u64 v[38:39], v[10:11], 0, v[18:19]             // 00000003AD6C: D2080026 0449010A
	flat_load_ushort v38, v[38:39]                             // 00000003AD74: DC480000 26000026
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003AD7C: BF8C0070
	v_lshlrev_b32_e32 v38, 16, v38                             // 00000003AD80: 244C4C90
	v_add_f32_e32 v20, v20, v38                                // 00000003AD84: 02284D14
	s_or_b64 exec, exec, s[18:19]                              // 00000003AD88: 87FE127E
	s_and_saveexec_b64 s[18:19], s[14:15]                      // 00000003AD8C: BE92200E
	s_cbranch_execz 7                                          // 00000003AD90: BF880007 <EpCombineIntraNodeKernel_bf16_nop2p+0x1db0>
	v_lshl_add_u64 v[38:39], v[12:13], 0, v[18:19]             // 00000003AD94: D2080026 0449010C
	flat_load_ushort v38, v[38:39]                             // 00000003AD9C: DC480000 26000026
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003ADA4: BF8C0070
	v_lshlrev_b32_e32 v38, 16, v38                             // 00000003ADA8: 244C4C90
	v_add_f32_e32 v20, v20, v38                                // 00000003ADAC: 02284D14
	s_or_b64 exec, exec, s[18:19]                              // 00000003ADB0: 87FE127E
	s_and_saveexec_b64 s[18:19], s[16:17]                      // 00000003ADB4: BE922010
	s_cbranch_execz 7                                          // 00000003ADB8: BF880007 <EpCombineIntraNodeKernel_bf16_nop2p+0x1dd8>
	v_lshl_add_u64 v[38:39], v[14:15], 0, v[18:19]             // 00000003ADBC: D2080026 0449010E
	flat_load_ushort v38, v[38:39]                             // 00000003ADC4: DC480000 26000026
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003ADCC: BF8C0070
	v_lshlrev_b32_e32 v38, 16, v38                             // 00000003ADD0: 244C4C90
	v_add_f32_e32 v20, v20, v38                                // 00000003ADD4: 02284D14
	s_or_b64 exec, exec, s[18:19]                              // 00000003ADD8: 87FE127E
	v_and_b32_e32 v38, 0x7f800000, v20                         // 00000003ADDC: 264C28FF 7F800000
	v_cmp_ne_u32_e64 s[18:19], s58, v38                        // 00000003ADE4: D0CD0012 00024C3A
	s_and_saveexec_b64 s[24:25], s[18:19]                      // 00000003ADEC: BE982012
	s_xor_b64 s[18:19], exec, s[24:25]                         // 00000003ADF0: 8892187E
	v_bfe_u32 v38, v20, 16, 1                                  // 00000003ADF4: D1C80026 02052114
	v_add3_u32 v38, v20, v38, s59                              // 00000003ADFC: D1FF0026 00EE4D14
	s_andn2_saveexec_b64 s[24:25], s[18:19]                    // 00000003AE04: BE982312
	s_cbranch_execz 65415                                      // 00000003AE08: BF88FF87 <EpCombineIntraNodeKernel_bf16_nop2p+0x1c28>
	v_or_b32_e32 v38, 0x10000, v20                             // 00000003AE0C: 284C28FF 00010000
	v_cmp_eq_u32_sdwa s[18:19], v20, v21 src0_sel:WORD_0 src1_sel:DWORD// 00000003AE14: 7D942AF9 06049214
	s_nop 1                                                    // 00000003AE1C: BF800001
	v_cndmask_b32_e64 v38, v38, v20, s[18:19]                  // 00000003AE20: D1000026 004A2926
	s_branch 65407                                             // 00000003AE28: BF82FF7F <EpCombineIntraNodeKernel_bf16_nop2p+0x1c28>
	s_or_b64 exec, exec, s[20:21]                              // 00000003AE2C: 87FE147E
	s_mov_b64 s[4:5], 0                                        // 00000003AE30: BE840180
	s_mov_b64 s[6:7], 0                                        // 00000003AE34: BE860180
	s_and_b64 vcc, exec, s[6:7]                                // 00000003AE38: 86EA067E
	s_cbranch_vccz 353                                         // 00000003AE3C: BF860161 <EpCombineIntraNodeKernel_bf16_nop2p+0x23c4>
	s_cmp_eq_u32 s90, 6                                        // 00000003AE40: BF06865A
	s_mov_b64 s[4:5], -1                                       // 00000003AE44: BE8401C1
	s_cbranch_scc0 350                                         // 00000003AE48: BF84015E <EpCombineIntraNodeKernel_bf16_nop2p+0x23c4>
	v_cmp_lt_u64_e32 vcc, s[84:85], v[36:37]                   // 00000003AE4C: 7DD24854
	v_mov_b64_e32 v[16:17], 0                                  // 00000003AE50: 7E207080
	s_and_saveexec_b64 s[28:29], vcc                           // 00000003AE54: BE9C206A
	s_cbranch_execz 211                                        // 00000003AE58: BF8800D3 <EpCombineIntraNodeKernel_bf16_nop2p+0x21a8>
	ds_read2_b64 v[8:11], v27 offset0:2 offset1:3              // 00000003AE5C: D8EE0302 0800001B
	ds_read2_b64 v[4:7], v27 offset1:1                         // 00000003AE64: D8EE0100 0400001B
	ds_read2_b64 v[12:15], v27 offset0:4 offset1:5             // 00000003AE6C: D8EE0504 0C00001B
	v_lshlrev_b32_e32 v20, 1, v26                              // 00000003AE74: 24283481
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[20:21]               // 00000003AE78: D2080000 04510122
	v_lshrrev_b64 v[2:3], 7, v[36:37]                          // 00000003AE80: D2900002 00024887
	s_mov_b64 s[30:31], 0                                      // 00000003AE88: BE9E0180
	s_waitcnt lgkmcnt(0)                                       // 00000003AE8C: BF8CC07F
	v_cmp_eq_u64_e32 vcc, 0, v[4:5]                            // 00000003AE90: 7DD40880
	v_cmp_ne_u64_e64 s[4:5], 0, v[4:5]                         // 00000003AE94: D0ED0004 00020880
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[20:21]                 // 00000003AE9C: D2080004 04510104
	v_cmp_eq_u64_e64 s[6:7], 0, v[6:7]                         // 00000003AEA4: D0EA0006 00020C80
	v_cmp_ne_u64_e64 s[8:9], 0, v[6:7]                         // 00000003AEAC: D0ED0008 00020C80
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[20:21]                 // 00000003AEB4: D2080006 04510106
	v_cmp_eq_u64_e64 s[10:11], 0, v[8:9]                       // 00000003AEBC: D0EA000A 00021080
	v_cmp_ne_u64_e64 s[12:13], 0, v[8:9]                       // 00000003AEC4: D0ED000C 00021080
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[20:21]                 // 00000003AECC: D2080008 04510108
	v_cmp_eq_u64_e64 s[14:15], 0, v[10:11]                     // 00000003AED4: D0EA000E 00021480
	v_cmp_ne_u64_e64 s[16:17], 0, v[10:11]                     // 00000003AEDC: D0ED0010 00021480
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[20:21]             // 00000003AEE4: D208000A 0451010A
	v_cmp_eq_u64_e64 s[18:19], 0, v[12:13]                     // 00000003AEEC: D0EA0012 00021880
	v_cmp_ne_u64_e64 s[20:21], 0, v[12:13]                     // 00000003AEF4: D0ED0014 00021880
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[20:21]             // 00000003AEFC: D208000C 0451010C
	v_cmp_eq_u64_e64 s[22:23], 0, v[14:15]                     // 00000003AF04: D0EA0016 00021C80
	v_cmp_ne_u64_e64 s[24:25], 0, v[14:15]                     // 00000003AF0C: D0ED0018 00021C80
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[20:21]             // 00000003AF14: D208000E 0451010E
	s_mov_b64 s[34:35], 0                                      // 00000003AF1C: BEA20180
	s_branch 31                                                // 00000003AF20: BF82001F <EpCombineIntraNodeKernel_bf16_nop2p+0x1fa0>
	s_or_b64 exec, exec, s[36:37]                              // 00000003AF24: 87FE247E
	v_lshrrev_b32_e32 v19, 16, v16                             // 00000003AF28: 20262090
	v_lshl_add_u64 v[16:17], s[34:35], 1, v[0:1]               // 00000003AF2C: D2080010 04010222
	s_add_u32 s34, s34, 0x80                                   // 00000003AF34: 8022FF22 00000080
	v_lshl_add_u64 v[2:3], v[2:3], 0, -1                       // 00000003AF3C: D2080002 03050102
	v_and_or_b32 v18, v18, s82, v19                            // 00000003AF44: D2010012 044CA512
	s_addc_u32 s35, s35, 0                                     // 00000003AF4C: 82238023
	v_cmp_eq_u64_e64 s[26:27], 0, v[2:3]                       // 00000003AF50: D0EA001A 00020480
	flat_store_dword v[16:17], v18 nt                          // 00000003AF58: DC720000 00001210
	v_lshl_add_u64 v[14:15], v[14:15], 0, s[88:89]             // 00000003AF60: D208000E 0161010E
	v_lshl_add_u64 v[12:13], v[12:13], 0, s[88:89]             // 00000003AF68: D208000C 0161010C
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[88:89]             // 00000003AF70: D208000A 0161010A
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[88:89]                 // 00000003AF78: D2080008 01610108
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[88:89]                 // 00000003AF80: D2080006 01610106
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[88:89]                 // 00000003AF88: D2080004 01610104
	s_or_b64 s[30:31], s[26:27], s[30:31]                      // 00000003AF90: 879E1E1A
	v_mov_b64_e32 v[16:17], s[34:35]                           // 00000003AF94: 7E207022
	s_andn2_b64 exec, exec, s[30:31]                           // 00000003AF98: 89FE1E7E
	s_cbranch_execz 129                                        // 00000003AF9C: BF880081 <EpCombineIntraNodeKernel_bf16_nop2p+0x21a4>
	s_and_saveexec_b64 s[26:27], s[4:5]                        // 00000003AFA0: BE9A2004
	s_cbranch_execz 2                                          // 00000003AFA4: BF880002 <EpCombineIntraNodeKernel_bf16_nop2p+0x1fb0>
	flat_load_dword v40, v[4:5] nt                             // 00000003AFA8: DC520000 28000004
	s_or_b64 exec, exec, s[26:27]                              // 00000003AFB0: 87FE1A7E
	s_and_saveexec_b64 s[26:27], s[8:9]                        // 00000003AFB4: BE9A2008
	s_cbranch_execz 2                                          // 00000003AFB8: BF880002 <EpCombineIntraNodeKernel_bf16_nop2p+0x1fc4>
	flat_load_dword v39, v[6:7] nt                             // 00000003AFBC: DC520000 27000006
	s_or_b64 exec, exec, s[26:27]                              // 00000003AFC4: 87FE1A7E
	s_and_saveexec_b64 s[26:27], s[12:13]                      // 00000003AFC8: BE9A200C
	s_cbranch_execz 2                                          // 00000003AFCC: BF880002 <EpCombineIntraNodeKernel_bf16_nop2p+0x1fd8>
	flat_load_dword v38, v[8:9] nt                             // 00000003AFD0: DC520000 26000008
	s_or_b64 exec, exec, s[26:27]                              // 00000003AFD8: 87FE1A7E
	s_and_saveexec_b64 s[26:27], s[16:17]                      // 00000003AFDC: BE9A2010
	s_cbranch_execz 2                                          // 00000003AFE0: BF880002 <EpCombineIntraNodeKernel_bf16_nop2p+0x1fec>
	flat_load_dword v20, v[10:11] nt                           // 00000003AFE4: DC520000 1400000A
	s_or_b64 exec, exec, s[26:27]                              // 00000003AFEC: 87FE1A7E
	s_and_saveexec_b64 s[26:27], s[20:21]                      // 00000003AFF0: BE9A2014
	s_cbranch_execz 2                                          // 00000003AFF4: BF880002 <EpCombineIntraNodeKernel_bf16_nop2p+0x2000>
	flat_load_dword v41, v[12:13] nt                           // 00000003AFF8: DC520000 2900000C
	s_or_b64 exec, exec, s[26:27]                              // 00000003B000: 87FE1A7E
	s_and_saveexec_b64 s[26:27], s[24:25]                      // 00000003B004: BE9A2018
	s_cbranch_execz 2                                          // 00000003B008: BF880002 <EpCombineIntraNodeKernel_bf16_nop2p+0x2014>
	flat_load_dword v44, v[14:15] nt                           // 00000003B00C: DC520000 2C00000E
	s_or_b64 exec, exec, s[26:27]                              // 00000003B014: 87FE1A7E
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003B018: BF8C0070
	v_lshlrev_b32_e32 v16, 16, v40                             // 00000003B01C: 24205090
	v_and_b32_e32 v17, 0xffff0000, v40                         // 00000003B020: 262250FF FFFF0000
	v_pk_add_f32 v[16:17], v[16:17], 0 op_sel_hi:[1,0]         // 00000003B028: D3B24010 08010110
	v_lshlrev_b32_e32 v18, 16, v39                             // 00000003B030: 24244E90
	v_cndmask_b32_e64 v17, v17, 0, vcc                         // 00000003B034: D1000011 01A90111
	v_cndmask_b32_e64 v16, v16, 0, vcc                         // 00000003B03C: D1000010 01A90110
	v_and_b32_e32 v19, 0xffff0000, v39                         // 00000003B044: 26264EFF FFFF0000
	v_pk_add_f32 v[18:19], v[16:17], v[18:19]                  // 00000003B04C: D3B24012 18022510
	s_nop 0                                                    // 00000003B054: BF800000
	v_cndmask_b32_e64 v17, v19, v17, s[6:7]                    // 00000003B058: D1000011 001A2313
	v_cndmask_b32_e64 v16, v18, v16, s[6:7]                    // 00000003B060: D1000010 001A2112
	v_lshlrev_b32_e32 v18, 16, v38                             // 00000003B068: 24244C90
	v_and_b32_e32 v19, 0xffff0000, v38                         // 00000003B06C: 26264CFF FFFF0000
	v_pk_add_f32 v[18:19], v[16:17], v[18:19]                  // 00000003B074: D3B24012 18022510
	s_nop 0                                                    // 00000003B07C: BF800000
	v_cndmask_b32_e64 v17, v19, v17, s[10:11]                  // 00000003B080: D1000011 002A2313
	v_cndmask_b32_e64 v16, v18, v16, s[10:11]                  // 00000003B088: D1000010 002A2112
	v_lshlrev_b32_e32 v18, 16, v20                             // 00000003B090: 24242890
	v_and_b32_e32 v19, 0xffff0000, v20                         // 00000003B094: 262628FF FFFF0000
	v_pk_add_f32 v[18:19], v[16:17], v[18:19]                  // 00000003B09C: D3B24012 18022510
	s_nop 0                                                    // 00000003B0A4: BF800000
	v_cndmask_b32_e64 v17, v19, v17, s[14:15]                  // 00000003B0A8: D1000011 003A2313
	v_cndmask_b32_e64 v16, v18, v16, s[14:15]                  // 00000003B0B0: D1000010 003A2112
	v_lshlrev_b32_e32 v18, 16, v41                             // 00000003B0B8: 24245290
	v_and_b32_e32 v19, 0xffff0000, v41                         // 00000003B0BC: 262652FF FFFF0000
	v_pk_add_f32 v[18:19], v[16:17], v[18:19]                  // 00000003B0C4: D3B24012 18022510
	s_nop 0                                                    // 00000003B0CC: BF800000
	v_cndmask_b32_e64 v17, v19, v17, s[18:19]                  // 00000003B0D0: D1000011 004A2313
	v_cndmask_b32_e64 v16, v18, v16, s[18:19]                  // 00000003B0D8: D1000010 004A2112
	v_lshlrev_b32_e32 v18, 16, v44                             // 00000003B0E0: 24245890
	v_and_b32_e32 v19, 0xffff0000, v44                         // 00000003B0E4: 262658FF FFFF0000
	v_pk_add_f32 v[18:19], v[16:17], v[18:19]                  // 00000003B0EC: D3B24012 18022510
	s_nop 0                                                    // 00000003B0F4: BF800000
	v_cndmask_b32_e64 v18, v18, v16, s[22:23]                  // 00000003B0F8: D1000012 005A2112
	v_and_b32_e32 v16, 0x7f800000, v18                         // 00000003B100: 262024FF 7F800000
	v_cmp_ne_u32_e64 s[26:27], s58, v16                        // 00000003B108: D0CD001A 0002203A
	s_and_saveexec_b64 s[36:37], s[26:27]                      // 00000003B110: BEA4201A
	s_xor_b64 s[26:27], exec, s[36:37]                         // 00000003B114: 889A247E
	v_bfe_u32 v16, v18, 16, 1                                  // 00000003B118: D1C80010 02052112
	v_add3_u32 v16, v18, v16, s59                              // 00000003B120: D1FF0010 00EE2112
	s_andn2_saveexec_b64 s[36:37], s[26:27]                    // 00000003B128: BEA4231A
	v_or_b32_e32 v16, 0x10000, v18                             // 00000003B12C: 282024FF 00010000
	v_cmp_eq_u32_sdwa s[26:27], v18, v21 src0_sel:WORD_0 src1_sel:DWORD// 00000003B134: 7D942AF9 06049A12
	s_nop 1                                                    // 00000003B13C: BF800001
	v_cndmask_b32_e64 v16, v16, v18, s[26:27]                  // 00000003B140: D1000010 006A2510
	s_or_b64 exec, exec, s[36:37]                              // 00000003B148: 87FE247E
	v_cndmask_b32_e64 v17, v19, v17, s[22:23]                  // 00000003B14C: D1000011 005A2313
	v_and_b32_e32 v18, 0x7f800000, v17                         // 00000003B154: 262422FF 7F800000
	v_cmp_ne_u32_e64 s[26:27], s58, v18                        // 00000003B15C: D0CD001A 0002243A
	s_and_saveexec_b64 s[36:37], s[26:27]                      // 00000003B164: BEA4201A
	s_xor_b64 s[26:27], exec, s[36:37]                         // 00000003B168: 889A247E
	v_bfe_u32 v18, v17, 16, 1                                  // 00000003B16C: D1C80012 02052111
	v_add3_u32 v18, v17, v18, s59                              // 00000003B174: D1FF0012 00EE2511
	s_andn2_saveexec_b64 s[36:37], s[26:27]                    // 00000003B17C: BEA4231A
	s_cbranch_execz 65384                                      // 00000003B180: BF88FF68 <EpCombineIntraNodeKernel_bf16_nop2p+0x1f24>
	v_or_b32_e32 v18, 0x10000, v17                             // 00000003B184: 282422FF 00010000
	v_cmp_eq_u32_sdwa s[26:27], v17, v21 src0_sel:WORD_0 src1_sel:DWORD// 00000003B18C: 7D942AF9 06049A11
	s_nop 1                                                    // 00000003B194: BF800001
	v_cndmask_b32_e64 v18, v18, v17, s[26:27]                  // 00000003B198: D1000012 006A2312
	s_branch 65376                                             // 00000003B1A0: BF82FF60 <EpCombineIntraNodeKernel_bf16_nop2p+0x1f24>
	s_or_b64 exec, exec, s[30:31]                              // 00000003B1A4: 87FE1E7E
	s_or_b64 exec, exec, s[28:29]                              // 00000003B1A8: 87FE1C7E
	v_lshl_add_u64 v[0:1], v[16:17], 0, v[42:43]               // 00000003B1AC: D2080000 04A90110
	v_cmp_lt_u64_e32 vcc, v[0:1], v[36:37]                     // 00000003B1B4: 7DD24900
	s_and_saveexec_b64 s[16:17], vcc                           // 00000003B1B8: BE90206A
	s_cbranch_execz 127                                        // 00000003B1BC: BF88007F <EpCombineIntraNodeKernel_bf16_nop2p+0x23bc>
	ds_read2_b64 v[12:15], v27 offset1:1                       // 00000003B1C0: D8EE0100 0C00001B
	ds_read2_b64 v[8:11], v27 offset0:2 offset1:3              // 00000003B1C8: D8EE0302 0800001B
	ds_read2_b64 v[4:7], v27 offset0:4 offset1:5               // 00000003B1D0: D8EE0504 0400001B
	v_lshlrev_b64 v[16:17], 1, v[0:1]                          // 00000003B1D8: D28F0010 00020081
	s_mov_b64 s[18:19], 0                                      // 00000003B1E0: BE920180
	s_waitcnt lgkmcnt(0)                                       // 00000003B1E4: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[12:13]                          // 00000003B1E8: 7DDA1880
	v_cmp_ne_u64_e64 s[4:5], 0, v[14:15]                       // 00000003B1EC: D0ED0004 00021C80
	v_cmp_ne_u64_e64 s[6:7], 0, v[8:9]                         // 00000003B1F4: D0ED0006 00021080
	v_cmp_ne_u64_e64 s[8:9], 0, v[10:11]                       // 00000003B1FC: D0ED0008 00021480
	v_cmp_ne_u64_e64 s[10:11], 0, v[4:5]                       // 00000003B204: D0ED000A 00020880
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]                       // 00000003B20C: D0ED000C 00020C80
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[16:17]                 // 00000003B214: D2080002 04410106
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[16:17]                 // 00000003B21C: D2080004 04410104
	v_lshl_add_u64 v[6:7], v[10:11], 0, v[16:17]               // 00000003B224: D2080006 0441010A
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[16:17]                 // 00000003B22C: D2080008 04410108
	v_lshl_add_u64 v[10:11], v[14:15], 0, v[16:17]             // 00000003B234: D208000A 0441010E
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[16:17]             // 00000003B23C: D208000C 0441010C
	s_branch 24                                                // 00000003B244: BF820018 <EpCombineIntraNodeKernel_bf16_nop2p+0x22a8>
	s_or_b64 exec, exec, s[20:21]                              // 00000003B248: 87FE147E
	v_lshl_add_u64 v[16:17], v[0:1], 1, v[34:35]               // 00000003B24C: D2080010 04890300
	v_lshl_add_u64 v[0:1], v[0:1], 0, 64                       // 00000003B254: D2080000 03010100
	v_cmp_ge_u64_e64 s[14:15], v[0:1], v[36:37]                // 00000003B25C: D0EE000E 00024900
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[86:87]                 // 00000003B264: D2080002 01590102
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[86:87]                 // 00000003B26C: D2080004 01590104
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[86:87]                 // 00000003B274: D2080006 01590106
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[86:87]                 // 00000003B27C: D2080008 01590108
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[86:87]             // 00000003B284: D208000A 0159010A
	s_or_b64 s[18:19], s[14:15], s[18:19]                      // 00000003B28C: 8792120E
	v_lshl_add_u64 v[12:13], v[12:13], 0, s[86:87]             // 00000003B290: D208000C 0159010C
	flat_store_short_d16_hi v[16:17], v15                      // 00000003B298: DC6C0000 00000F10
	s_andn2_b64 exec, exec, s[18:19]                           // 00000003B2A0: 89FE127E
	s_cbranch_execz 69                                         // 00000003B2A4: BF880045 <EpCombineIntraNodeKernel_bf16_nop2p+0x23bc>
	v_mov_b32_e32 v14, 0                                       // 00000003B2A8: 7E1C0280
	s_and_saveexec_b64 s[14:15], vcc                           // 00000003B2AC: BE8E206A
	s_cbranch_execz 5                                          // 00000003B2B0: BF880005 <EpCombineIntraNodeKernel_bf16_nop2p+0x22c8>
	flat_load_ushort v14, v[12:13]                             // 00000003B2B4: DC480000 0E00000C
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003B2BC: BF8C0070
	v_lshlrev_b32_e32 v14, 16, v14                             // 00000003B2C0: 241C1C90
	v_add_f32_e32 v14, 0, v14                                  // 00000003B2C4: 021C1C80
	s_or_b64 exec, exec, s[14:15]                              // 00000003B2C8: 87FE0E7E
	s_and_saveexec_b64 s[14:15], s[4:5]                        // 00000003B2CC: BE8E2004
	s_cbranch_execz 5                                          // 00000003B2D0: BF880005 <EpCombineIntraNodeKernel_bf16_nop2p+0x22e8>
	flat_load_ushort v15, v[10:11]                             // 00000003B2D4: DC480000 0F00000A
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003B2DC: BF8C0070
	v_lshlrev_b32_e32 v15, 16, v15                             // 00000003B2E0: 241E1E90
	v_add_f32_e32 v14, v14, v15                                // 00000003B2E4: 021C1F0E
	s_or_b64 exec, exec, s[14:15]                              // 00000003B2E8: 87FE0E7E
	s_and_saveexec_b64 s[14:15], s[6:7]                        // 00000003B2EC: BE8E2006
	s_cbranch_execz 5                                          // 00000003B2F0: BF880005 <EpCombineIntraNodeKernel_bf16_nop2p+0x2308>
	flat_load_ushort v15, v[8:9]                               // 00000003B2F4: DC480000 0F000008
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003B2FC: BF8C0070
	v_lshlrev_b32_e32 v15, 16, v15                             // 00000003B300: 241E1E90
	v_add_f32_e32 v14, v14, v15                                // 00000003B304: 021C1F0E
	s_or_b64 exec, exec, s[14:15]                              // 00000003B308: 87FE0E7E
	s_and_saveexec_b64 s[14:15], s[8:9]                        // 00000003B30C: BE8E2008
	s_cbranch_execz 5                                          // 00000003B310: BF880005 <EpCombineIntraNodeKernel_bf16_nop2p+0x2328>
	flat_load_ushort v15, v[6:7]                               // 00000003B314: DC480000 0F000006
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003B31C: BF8C0070
	v_lshlrev_b32_e32 v15, 16, v15                             // 00000003B320: 241E1E90
	v_add_f32_e32 v14, v14, v15                                // 00000003B324: 021C1F0E
	s_or_b64 exec, exec, s[14:15]                              // 00000003B328: 87FE0E7E
	s_and_saveexec_b64 s[14:15], s[10:11]                      // 00000003B32C: BE8E200A
	s_cbranch_execz 5                                          // 00000003B330: BF880005 <EpCombineIntraNodeKernel_bf16_nop2p+0x2348>
	flat_load_ushort v15, v[4:5]                               // 00000003B334: DC480000 0F000004
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003B33C: BF8C0070
	v_lshlrev_b32_e32 v15, 16, v15                             // 00000003B340: 241E1E90
	v_add_f32_e32 v14, v14, v15                                // 00000003B344: 021C1F0E
	s_or_b64 exec, exec, s[14:15]                              // 00000003B348: 87FE0E7E
	s_and_saveexec_b64 s[14:15], s[12:13]                      // 00000003B34C: BE8E200C
	s_cbranch_execz 5                                          // 00000003B350: BF880005 <EpCombineIntraNodeKernel_bf16_nop2p+0x2368>
	flat_load_ushort v15, v[2:3]                               // 00000003B354: DC480000 0F000002
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003B35C: BF8C0070
	v_lshlrev_b32_e32 v15, 16, v15                             // 00000003B360: 241E1E90
	v_add_f32_e32 v14, v14, v15                                // 00000003B364: 021C1F0E
	s_or_b64 exec, exec, s[14:15]                              // 00000003B368: 87FE0E7E
	v_and_b32_e32 v15, 0x7f800000, v14                         // 00000003B36C: 261E1CFF 7F800000
	v_cmp_ne_u32_e64 s[14:15], s58, v15                        // 00000003B374: D0CD000E 00021E3A
	s_and_saveexec_b64 s[20:21], s[14:15]                      // 00000003B37C: BE94200E
	s_xor_b64 s[14:15], exec, s[20:21]                         // 00000003B380: 888E147E
	v_bfe_u32 v15, v14, 16, 1                                  // 00000003B384: D1C8000F 0205210E
	v_add3_u32 v15, v14, v15, s59                              // 00000003B38C: D1FF000F 00EE1F0E
	s_andn2_saveexec_b64 s[20:21], s[14:15]                    // 00000003B394: BE94230E
	s_cbranch_execz 65451                                      // 00000003B398: BF88FFAB <EpCombineIntraNodeKernel_bf16_nop2p+0x2248>
	v_or_b32_e32 v15, 0x10000, v14                             // 00000003B39C: 281E1CFF 00010000
	v_cmp_eq_u32_sdwa s[14:15], v14, v21 src0_sel:WORD_0 src1_sel:DWORD// 00000003B3A4: 7D942AF9 06048E0E
	s_nop 1                                                    // 00000003B3AC: BF800001
	v_cndmask_b32_e64 v15, v15, v14, s[14:15]                  // 00000003B3B0: D100000F 003A1D0F
	s_branch 65443                                             // 00000003B3B8: BF82FFA3 <EpCombineIntraNodeKernel_bf16_nop2p+0x2248>
	s_or_b64 exec, exec, s[16:17]                              // 00000003B3BC: 87FE107E
	s_mov_b64 s[4:5], 0                                        // 00000003B3C0: BE840180
	s_mov_b64 s[6:7], 0                                        // 00000003B3C4: BE860180
	s_and_b64 vcc, exec, s[6:7]                                // 00000003B3C8: 86EA067E
	s_cbranch_vccz 487                                         // 00000003B3CC: BF8601E7 <EpCombineIntraNodeKernel_bf16_nop2p+0x2b6c>
	s_cmp_gt_i32 s90, 1                                        // 00000003B3D0: BF02815A
	s_mov_b64 s[6:7], -1                                       // 00000003B3D4: BE8601C1
	s_cbranch_scc0 478                                         // 00000003B3D8: BF8401DE <EpCombineIntraNodeKernel_bf16_nop2p+0x2b54>
	s_cmp_gt_i32 s90, 3                                        // 00000003B3DC: BF02835A
	s_cbranch_scc0 276                                         // 00000003B3E0: BF840114 <EpCombineIntraNodeKernel_bf16_nop2p+0x2834>
	s_cmp_eq_u32 s90, 4                                        // 00000003B3E4: BF06845A
	s_mov_b64 s[4:5], -1                                       // 00000003B3E8: BE8401C1
	s_cbranch_scc0 272                                         // 00000003B3EC: BF840110 <EpCombineIntraNodeKernel_bf16_nop2p+0x2830>
	v_cmp_lt_u64_e32 vcc, s[84:85], v[36:37]                   // 00000003B3F0: 7DD24854
	v_mov_b64_e32 v[12:13], 0                                  // 00000003B3F4: 7E187080
	s_and_saveexec_b64 s[20:21], vcc                           // 00000003B3F8: BE94206A
	s_cbranch_execz 163                                        // 00000003B3FC: BF8800A3 <EpCombineIntraNodeKernel_bf16_nop2p+0x268c>
	ds_read2_b64 v[8:11], v27 offset0:2 offset1:3              // 00000003B400: D8EE0302 0800001B
	ds_read2_b64 v[4:7], v27 offset1:1                         // 00000003B408: D8EE0100 0400001B
	v_lshlrev_b32_e32 v20, 1, v26                              // 00000003B410: 24283481
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[20:21]               // 00000003B414: D2080000 04510122
	v_lshrrev_b64 v[2:3], 7, v[36:37]                          // 00000003B41C: D2900002 00024887
	s_mov_b64 s[22:23], 0                                      // 00000003B424: BE960180
	s_waitcnt lgkmcnt(0)                                       // 00000003B428: BF8CC07F
	v_cmp_eq_u64_e32 vcc, 0, v[4:5]                            // 00000003B42C: 7DD40880
	v_cmp_ne_u64_e64 s[4:5], 0, v[4:5]                         // 00000003B430: D0ED0004 00020880
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[20:21]                 // 00000003B438: D2080004 04510104
	v_cmp_eq_u64_e64 s[6:7], 0, v[6:7]                         // 00000003B440: D0EA0006 00020C80
	v_cmp_ne_u64_e64 s[8:9], 0, v[6:7]                         // 00000003B448: D0ED0008 00020C80
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[20:21]                 // 00000003B450: D2080006 04510106
	v_cmp_eq_u64_e64 s[10:11], 0, v[8:9]                       // 00000003B458: D0EA000A 00021080
	v_cmp_ne_u64_e64 s[12:13], 0, v[8:9]                       // 00000003B460: D0ED000C 00021080
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[20:21]                 // 00000003B468: D2080008 04510108
	v_cmp_eq_u64_e64 s[14:15], 0, v[10:11]                     // 00000003B470: D0EA000E 00021480
	v_cmp_ne_u64_e64 s[16:17], 0, v[10:11]                     // 00000003B478: D0ED0010 00021480
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[20:21]             // 00000003B480: D208000A 0451010A
	s_mov_b64 s[24:25], 0                                      // 00000003B488: BE980180
	s_branch 27                                                // 00000003B48C: BF82001B <EpCombineIntraNodeKernel_bf16_nop2p+0x24fc>
	s_or_b64 exec, exec, s[26:27]                              // 00000003B490: 87FE1A7E
	v_lshrrev_b32_e32 v15, 16, v12                             // 00000003B494: 201E1890
	v_lshl_add_u64 v[12:13], s[24:25], 1, v[0:1]               // 00000003B498: D208000C 04010218
	s_add_u32 s24, s24, 0x80                                   // 00000003B4A0: 8018FF18 00000080
	v_lshl_add_u64 v[2:3], v[2:3], 0, -1                       // 00000003B4A8: D2080002 03050102
	v_and_or_b32 v14, v14, s82, v15                            // 00000003B4B0: D201000E 043CA50E
	s_addc_u32 s25, s25, 0                                     // 00000003B4B8: 82198019
	v_cmp_eq_u64_e64 s[18:19], 0, v[2:3]                       // 00000003B4BC: D0EA0012 00020480
	flat_store_dword v[12:13], v14 nt                          // 00000003B4C4: DC720000 00000E0C
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[88:89]             // 00000003B4CC: D208000A 0161010A
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[88:89]                 // 00000003B4D4: D2080008 01610108
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[88:89]                 // 00000003B4DC: D2080006 01610106
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[88:89]                 // 00000003B4E4: D2080004 01610104
	s_or_b64 s[22:23], s[18:19], s[22:23]                      // 00000003B4EC: 87961612
	v_mov_b64_e32 v[12:13], s[24:25]                           // 00000003B4F0: 7E187018
	s_andn2_b64 exec, exec, s[22:23]                           // 00000003B4F4: 89FE167E
	s_cbranch_execz 99                                         // 00000003B4F8: BF880063 <EpCombineIntraNodeKernel_bf16_nop2p+0x2688>
	s_and_saveexec_b64 s[18:19], s[4:5]                        // 00000003B4FC: BE922004
	s_cbranch_execz 2                                          // 00000003B500: BF880002 <EpCombineIntraNodeKernel_bf16_nop2p+0x250c>
	flat_load_dword v17, v[4:5] nt                             // 00000003B504: DC520000 11000004
	s_or_b64 exec, exec, s[18:19]                              // 00000003B50C: 87FE127E
	s_and_saveexec_b64 s[18:19], s[8:9]                        // 00000003B510: BE922008
	s_cbranch_execz 2                                          // 00000003B514: BF880002 <EpCombineIntraNodeKernel_bf16_nop2p+0x2520>
	flat_load_dword v16, v[6:7] nt                             // 00000003B518: DC520000 10000006
	s_or_b64 exec, exec, s[18:19]                              // 00000003B520: 87FE127E
	s_and_saveexec_b64 s[18:19], s[12:13]                      // 00000003B524: BE92200C
	s_cbranch_execz 2                                          // 00000003B528: BF880002 <EpCombineIntraNodeKernel_bf16_nop2p+0x2534>
	flat_load_dword v18, v[8:9] nt                             // 00000003B52C: DC520000 12000008
	s_or_b64 exec, exec, s[18:19]                              // 00000003B534: 87FE127E
	s_and_saveexec_b64 s[18:19], s[16:17]                      // 00000003B538: BE922010
	s_cbranch_execz 2                                          // 00000003B53C: BF880002 <EpCombineIntraNodeKernel_bf16_nop2p+0x2548>
	flat_load_dword v19, v[10:11] nt                           // 00000003B540: DC520000 1300000A
	s_or_b64 exec, exec, s[18:19]                              // 00000003B548: 87FE127E
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003B54C: BF8C0070
	v_lshlrev_b32_e32 v12, 16, v17                             // 00000003B550: 24182290
	v_and_b32_e32 v13, 0xffff0000, v17                         // 00000003B554: 261A22FF FFFF0000
	v_pk_add_f32 v[12:13], v[12:13], 0 op_sel_hi:[1,0]         // 00000003B55C: D3B2400C 0801010C
	v_lshlrev_b32_e32 v14, 16, v16                             // 00000003B564: 241C2090
	v_cndmask_b32_e64 v13, v13, 0, vcc                         // 00000003B568: D100000D 01A9010D
	v_cndmask_b32_e64 v12, v12, 0, vcc                         // 00000003B570: D100000C 01A9010C
	v_and_b32_e32 v15, 0xffff0000, v16                         // 00000003B578: 261E20FF FFFF0000
	v_pk_add_f32 v[14:15], v[12:13], v[14:15]                  // 00000003B580: D3B2400E 18021D0C
	s_nop 0                                                    // 00000003B588: BF800000
	v_cndmask_b32_e64 v13, v15, v13, s[6:7]                    // 00000003B58C: D100000D 001A1B0F
	v_cndmask_b32_e64 v12, v14, v12, s[6:7]                    // 00000003B594: D100000C 001A190E
	v_lshlrev_b32_e32 v14, 16, v18                             // 00000003B59C: 241C2490
	v_and_b32_e32 v15, 0xffff0000, v18                         // 00000003B5A0: 261E24FF FFFF0000
	v_pk_add_f32 v[14:15], v[12:13], v[14:15]                  // 00000003B5A8: D3B2400E 18021D0C
	s_nop 0                                                    // 00000003B5B0: BF800000
	v_cndmask_b32_e64 v13, v15, v13, s[10:11]                  // 00000003B5B4: D100000D 002A1B0F
	v_cndmask_b32_e64 v12, v14, v12, s[10:11]                  // 00000003B5BC: D100000C 002A190E
	v_lshlrev_b32_e32 v14, 16, v19                             // 00000003B5C4: 241C2690
	v_and_b32_e32 v15, 0xffff0000, v19                         // 00000003B5C8: 261E26FF FFFF0000
	v_pk_add_f32 v[14:15], v[12:13], v[14:15]                  // 00000003B5D0: D3B2400E 18021D0C
	s_nop 0                                                    // 00000003B5D8: BF800000
	v_cndmask_b32_e64 v14, v14, v12, s[14:15]                  // 00000003B5DC: D100000E 003A190E
	v_and_b32_e32 v12, 0x7f800000, v14                         // 00000003B5E4: 26181CFF 7F800000
	v_cmp_ne_u32_e64 s[18:19], s58, v12                        // 00000003B5EC: D0CD0012 0002183A
	s_and_saveexec_b64 s[26:27], s[18:19]                      // 00000003B5F4: BE9A2012
	s_xor_b64 s[18:19], exec, s[26:27]                         // 00000003B5F8: 88921A7E
	v_bfe_u32 v12, v14, 16, 1                                  // 00000003B5FC: D1C8000C 0205210E
	v_add3_u32 v12, v14, v12, s59                              // 00000003B604: D1FF000C 00EE190E
	s_andn2_saveexec_b64 s[26:27], s[18:19]                    // 00000003B60C: BE9A2312
	v_or_b32_e32 v12, 0x10000, v14                             // 00000003B610: 28181CFF 00010000
	v_cmp_eq_u32_sdwa s[18:19], v14, v21 src0_sel:WORD_0 src1_sel:DWORD// 00000003B618: 7D942AF9 0604920E
	s_nop 1                                                    // 00000003B620: BF800001
	v_cndmask_b32_e64 v12, v12, v14, s[18:19]                  // 00000003B624: D100000C 004A1D0C
	s_or_b64 exec, exec, s[26:27]                              // 00000003B62C: 87FE1A7E
	v_cndmask_b32_e64 v13, v15, v13, s[14:15]                  // 00000003B630: D100000D 003A1B0F
	v_and_b32_e32 v14, 0x7f800000, v13                         // 00000003B638: 261C1AFF 7F800000
	v_cmp_ne_u32_e64 s[18:19], s58, v14                        // 00000003B640: D0CD0012 00021C3A
	s_and_saveexec_b64 s[26:27], s[18:19]                      // 00000003B648: BE9A2012
	s_xor_b64 s[18:19], exec, s[26:27]                         // 00000003B64C: 88921A7E
	v_bfe_u32 v14, v13, 16, 1                                  // 00000003B650: D1C8000E 0205210D
	v_add3_u32 v14, v13, v14, s59                              // 00000003B658: D1FF000E 00EE1D0D
	s_andn2_saveexec_b64 s[26:27], s[18:19]                    // 00000003B660: BE9A2312
	s_cbranch_execz 65418                                      // 00000003B664: BF88FF8A <EpCombineIntraNodeKernel_bf16_nop2p+0x2490>
	v_or_b32_e32 v14, 0x10000, v13                             // 00000003B668: 281C1AFF 00010000
	v_cmp_eq_u32_sdwa s[18:19], v13, v21 src0_sel:WORD_0 src1_sel:DWORD// 00000003B670: 7D942AF9 0604920D
	s_nop 1                                                    // 00000003B678: BF800001
	v_cndmask_b32_e64 v14, v14, v13, s[18:19]                  // 00000003B67C: D100000E 004A1B0E
	s_branch 65410                                             // 00000003B684: BF82FF82 <EpCombineIntraNodeKernel_bf16_nop2p+0x2490>
	s_or_b64 exec, exec, s[22:23]                              // 00000003B688: 87FE167E
	s_or_b64 exec, exec, s[20:21]                              // 00000003B68C: 87FE147E
	v_lshl_add_u64 v[0:1], v[12:13], 0, v[42:43]               // 00000003B690: D2080000 04A9010C
	v_cmp_lt_u64_e32 vcc, v[0:1], v[36:37]                     // 00000003B698: 7DD24900
	s_and_saveexec_b64 s[12:13], vcc                           // 00000003B69C: BE8C206A
	s_cbranch_execz 97                                         // 00000003B6A0: BF880061 <EpCombineIntraNodeKernel_bf16_nop2p+0x2828>
	ds_read2_b64 v[8:11], v27 offset1:1                        // 00000003B6A4: D8EE0100 0800001B
	ds_read2_b64 v[4:7], v27 offset0:2 offset1:3               // 00000003B6AC: D8EE0302 0400001B
	v_lshlrev_b64 v[12:13], 1, v[0:1]                          // 00000003B6B4: D28F000C 00020081
	s_mov_b64 s[14:15], 0                                      // 00000003B6BC: BE8E0180
	s_waitcnt lgkmcnt(0)                                       // 00000003B6C0: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[8:9]                            // 00000003B6C4: 7DDA1080
	v_cmp_ne_u64_e64 s[4:5], 0, v[10:11]                       // 00000003B6C8: D0ED0004 00021480
	v_cmp_ne_u64_e64 s[6:7], 0, v[4:5]                         // 00000003B6D0: D0ED0006 00020880
	v_cmp_ne_u64_e64 s[8:9], 0, v[6:7]                         // 00000003B6D8: D0ED0008 00020C80
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[12:13]                 // 00000003B6E0: D2080002 04310106
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[12:13]                 // 00000003B6E8: D2080004 04310104
	v_lshl_add_u64 v[6:7], v[10:11], 0, v[12:13]               // 00000003B6F0: D2080006 0431010A
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[12:13]                 // 00000003B6F8: D2080008 04310108
	s_branch 20                                                // 00000003B700: BF820014 <EpCombineIntraNodeKernel_bf16_nop2p+0x2754>
	s_or_b64 exec, exec, s[16:17]                              // 00000003B704: 87FE107E
	v_lshl_add_u64 v[12:13], v[0:1], 1, v[34:35]               // 00000003B708: D208000C 04890300
	v_lshl_add_u64 v[0:1], v[0:1], 0, 64                       // 00000003B710: D2080000 03010100
	v_cmp_ge_u64_e64 s[10:11], v[0:1], v[36:37]                // 00000003B718: D0EE000A 00024900
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[86:87]                 // 00000003B720: D2080002 01590102
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[86:87]                 // 00000003B728: D2080004 01590104
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[86:87]                 // 00000003B730: D2080006 01590106
	s_or_b64 s[14:15], s[10:11], s[14:15]                      // 00000003B738: 878E0E0A
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[86:87]                 // 00000003B73C: D2080008 01590108
	flat_store_short_d16_hi v[12:13], v11                      // 00000003B744: DC6C0000 00000B0C
	s_andn2_b64 exec, exec, s[14:15]                           // 00000003B74C: 89FE0E7E
	s_cbranch_execz 53                                         // 00000003B750: BF880035 <EpCombineIntraNodeKernel_bf16_nop2p+0x2828>
	v_mov_b32_e32 v10, 0                                       // 00000003B754: 7E140280
	s_and_saveexec_b64 s[10:11], vcc                           // 00000003B758: BE8A206A
	s_cbranch_execz 5                                          // 00000003B75C: BF880005 <EpCombineIntraNodeKernel_bf16_nop2p+0x2774>
	flat_load_ushort v10, v[8:9]                               // 00000003B760: DC480000 0A000008
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003B768: BF8C0070
	v_lshlrev_b32_e32 v10, 16, v10                             // 00000003B76C: 24141490
	v_add_f32_e32 v10, 0, v10                                  // 00000003B770: 02141480
	s_or_b64 exec, exec, s[10:11]                              // 00000003B774: 87FE0A7E
	s_and_saveexec_b64 s[10:11], s[4:5]                        // 00000003B778: BE8A2004
	s_cbranch_execz 5                                          // 00000003B77C: BF880005 <EpCombineIntraNodeKernel_bf16_nop2p+0x2794>
	flat_load_ushort v11, v[6:7]                               // 00000003B780: DC480000 0B000006
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003B788: BF8C0070
	v_lshlrev_b32_e32 v11, 16, v11                             // 00000003B78C: 24161690
	v_add_f32_e32 v10, v10, v11                                // 00000003B790: 0214170A
	s_or_b64 exec, exec, s[10:11]                              // 00000003B794: 87FE0A7E
	s_and_saveexec_b64 s[10:11], s[6:7]                        // 00000003B798: BE8A2006
	s_cbranch_execz 5                                          // 00000003B79C: BF880005 <EpCombineIntraNodeKernel_bf16_nop2p+0x27b4>
	flat_load_ushort v11, v[4:5]                               // 00000003B7A0: DC480000 0B000004
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003B7A8: BF8C0070
	v_lshlrev_b32_e32 v11, 16, v11                             // 00000003B7AC: 24161690
	v_add_f32_e32 v10, v10, v11                                // 00000003B7B0: 0214170A
	s_or_b64 exec, exec, s[10:11]                              // 00000003B7B4: 87FE0A7E
	s_and_saveexec_b64 s[10:11], s[8:9]                        // 00000003B7B8: BE8A2008
	s_cbranch_execz 5                                          // 00000003B7BC: BF880005 <EpCombineIntraNodeKernel_bf16_nop2p+0x27d4>
	flat_load_ushort v11, v[2:3]                               // 00000003B7C0: DC480000 0B000002
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003B7C8: BF8C0070
	v_lshlrev_b32_e32 v11, 16, v11                             // 00000003B7CC: 24161690
	v_add_f32_e32 v10, v10, v11                                // 00000003B7D0: 0214170A
	s_or_b64 exec, exec, s[10:11]                              // 00000003B7D4: 87FE0A7E
	v_and_b32_e32 v11, 0x7f800000, v10                         // 00000003B7D8: 261614FF 7F800000
	v_cmp_ne_u32_e64 s[10:11], s58, v11                        // 00000003B7E0: D0CD000A 0002163A
	s_and_saveexec_b64 s[16:17], s[10:11]                      // 00000003B7E8: BE90200A
	s_xor_b64 s[10:11], exec, s[16:17]                         // 00000003B7EC: 888A107E
	v_bfe_u32 v11, v10, 16, 1                                  // 00000003B7F0: D1C8000B 0205210A
	v_add3_u32 v11, v10, v11, s59                              // 00000003B7F8: D1FF000B 00EE170A
	s_andn2_saveexec_b64 s[16:17], s[10:11]                    // 00000003B800: BE90230A
	s_cbranch_execz 65471                                      // 00000003B804: BF88FFBF <EpCombineIntraNodeKernel_bf16_nop2p+0x2704>
	v_or_b32_e32 v11, 0x10000, v10                             // 00000003B808: 281614FF 00010000
	v_cmp_eq_u32_sdwa s[10:11], v10, v21 src0_sel:WORD_0 src1_sel:DWORD// 00000003B810: 7D942AF9 06048A0A
	s_nop 1                                                    // 00000003B818: BF800001
	v_cndmask_b32_e64 v11, v11, v10, s[10:11]                  // 00000003B81C: D100000B 002A150B
	s_branch 65463                                             // 00000003B824: BF82FFB7 <EpCombineIntraNodeKernel_bf16_nop2p+0x2704>
	s_or_b64 exec, exec, s[12:13]                              // 00000003B828: 87FE0C7E
	s_mov_b64 s[4:5], 0                                        // 00000003B82C: BE840180
	s_mov_b64 s[6:7], 0                                        // 00000003B830: BE860180
	s_and_b64 vcc, exec, s[6:7]                                // 00000003B834: 86EA067E
	s_cbranch_vccz 197                                         // 00000003B838: BF8600C5 <EpCombineIntraNodeKernel_bf16_nop2p+0x2b50>
	s_cmp_eq_u32 s90, 2                                        // 00000003B83C: BF06825A
	s_mov_b64 s[4:5], -1                                       // 00000003B840: BE8401C1
	s_cbranch_scc0 194                                         // 00000003B844: BF8400C2 <EpCombineIntraNodeKernel_bf16_nop2p+0x2b50>
	v_cmp_lt_u64_e32 vcc, s[84:85], v[36:37]                   // 00000003B848: 7DD24854
	v_mov_b64_e32 v[8:9], 0                                    // 00000003B84C: 7E107080
	s_and_saveexec_b64 s[12:13], vcc                           // 00000003B850: BE8C206A
	s_cbranch_execz 115                                        // 00000003B854: BF880073 <EpCombineIntraNodeKernel_bf16_nop2p+0x2a24>
	ds_read2_b64 v[4:7], v27 offset1:1                         // 00000003B858: D8EE0100 0400001B
	v_lshlrev_b32_e32 v20, 1, v26                              // 00000003B860: 24283481
	v_lshrrev_b64 v[0:1], 7, v[36:37]                          // 00000003B864: D2900000 00024887
	s_mov_b64 s[14:15], 0                                      // 00000003B86C: BE8E0180
	v_lshl_add_u64 v[2:3], v[34:35], 0, v[20:21]               // 00000003B870: D2080002 04510122
	s_waitcnt lgkmcnt(0)                                       // 00000003B878: BF8CC07F
	v_cmp_eq_u64_e32 vcc, 0, v[4:5]                            // 00000003B87C: 7DD40880
	v_cmp_ne_u64_e64 s[4:5], 0, v[4:5]                         // 00000003B880: D0ED0004 00020880
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[20:21]                 // 00000003B888: D2080004 04510104
	v_cmp_eq_u64_e64 s[6:7], 0, v[6:7]                         // 00000003B890: D0EA0006 00020C80
	v_cmp_ne_u64_e64 s[8:9], 0, v[6:7]                         // 00000003B898: D0ED0008 00020C80
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[20:21]                 // 00000003B8A0: D2080006 04510106
	s_mov_b64 s[16:17], 0                                      // 00000003B8A8: BE900180
	s_branch 23                                                // 00000003B8AC: BF820017 <EpCombineIntraNodeKernel_bf16_nop2p+0x290c>
	s_or_b64 exec, exec, s[18:19]                              // 00000003B8B0: 87FE127E
	v_lshrrev_b32_e32 v11, 16, v8                              // 00000003B8B4: 20161090
	v_lshl_add_u64 v[8:9], s[16:17], 1, v[2:3]                 // 00000003B8B8: D2080008 04090210
	s_add_u32 s16, s16, 0x80                                   // 00000003B8C0: 8010FF10 00000080
	v_lshl_add_u64 v[0:1], v[0:1], 0, -1                       // 00000003B8C8: D2080000 03050100
	v_and_or_b32 v10, v10, s82, v11                            // 00000003B8D0: D201000A 042CA50A
	s_addc_u32 s17, s17, 0                                     // 00000003B8D8: 82118011
	v_cmp_eq_u64_e64 s[10:11], 0, v[0:1]                       // 00000003B8DC: D0EA000A 00020080
	flat_store_dword v[8:9], v10 nt                            // 00000003B8E4: DC720000 00000A08
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[88:89]                 // 00000003B8EC: D2080006 01610106
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[88:89]                 // 00000003B8F4: D2080004 01610104
	s_or_b64 s[14:15], s[10:11], s[14:15]                      // 00000003B8FC: 878E0E0A
	v_mov_b64_e32 v[8:9], s[16:17]                             // 00000003B900: 7E107010
	s_andn2_b64 exec, exec, s[14:15]                           // 00000003B904: 89FE0E7E
	s_cbranch_execz 69                                         // 00000003B908: BF880045 <EpCombineIntraNodeKernel_bf16_nop2p+0x2a20>
	s_and_saveexec_b64 s[10:11], s[4:5]                        // 00000003B90C: BE8A2004
	s_cbranch_execz 2                                          // 00000003B910: BF880002 <EpCombineIntraNodeKernel_bf16_nop2p+0x291c>
	flat_load_dword v12, v[4:5] nt                             // 00000003B914: DC520000 0C000004
	s_or_b64 exec, exec, s[10:11]                              // 00000003B91C: 87FE0A7E
	s_and_saveexec_b64 s[10:11], s[8:9]                        // 00000003B920: BE8A2008
	s_cbranch_execz 2                                          // 00000003B924: BF880002 <EpCombineIntraNodeKernel_bf16_nop2p+0x2930>
	flat_load_dword v13, v[6:7] nt                             // 00000003B928: DC520000 0D000006
	s_or_b64 exec, exec, s[10:11]                              // 00000003B930: 87FE0A7E
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003B934: BF8C0070
	v_lshlrev_b32_e32 v8, 16, v12                              // 00000003B938: 24101890
	v_and_b32_e32 v9, 0xffff0000, v12                          // 00000003B93C: 261218FF FFFF0000
	v_pk_add_f32 v[8:9], v[8:9], 0 op_sel_hi:[1,0]             // 00000003B944: D3B24008 08010108
	v_lshlrev_b32_e32 v10, 16, v13                             // 00000003B94C: 24141A90
	v_cndmask_b32_e64 v9, v9, 0, vcc                           // 00000003B950: D1000009 01A90109
	v_cndmask_b32_e64 v8, v8, 0, vcc                           // 00000003B958: D1000008 01A90108
	v_and_b32_e32 v11, 0xffff0000, v13                         // 00000003B960: 26161AFF FFFF0000
	v_pk_add_f32 v[10:11], v[8:9], v[10:11]                    // 00000003B968: D3B2400A 18021508
	s_nop 0                                                    // 00000003B970: BF800000
	v_cndmask_b32_e64 v10, v10, v8, s[6:7]                     // 00000003B974: D100000A 001A110A
	v_and_b32_e32 v8, 0x7f800000, v10                          // 00000003B97C: 261014FF 7F800000
	v_cmp_ne_u32_e64 s[10:11], s58, v8                         // 00000003B984: D0CD000A 0002103A
	s_and_saveexec_b64 s[18:19], s[10:11]                      // 00000003B98C: BE92200A
	s_xor_b64 s[10:11], exec, s[18:19]                         // 00000003B990: 888A127E
	v_bfe_u32 v8, v10, 16, 1                                   // 00000003B994: D1C80008 0205210A
	v_add3_u32 v8, v10, v8, s59                                // 00000003B99C: D1FF0008 00EE110A
	s_andn2_saveexec_b64 s[18:19], s[10:11]                    // 00000003B9A4: BE92230A
	v_or_b32_e32 v8, 0x10000, v10                              // 00000003B9A8: 281014FF 00010000
	v_cmp_eq_u32_sdwa s[10:11], v10, v21 src0_sel:WORD_0 src1_sel:DWORD// 00000003B9B0: 7D942AF9 06048A0A
	s_nop 1                                                    // 00000003B9B8: BF800001
	v_cndmask_b32_e64 v8, v8, v10, s[10:11]                    // 00000003B9BC: D1000008 002A1508
	s_or_b64 exec, exec, s[18:19]                              // 00000003B9C4: 87FE127E
	v_cndmask_b32_e64 v9, v11, v9, s[6:7]                      // 00000003B9C8: D1000009 001A130B
	v_and_b32_e32 v10, 0x7f800000, v9                          // 00000003B9D0: 261412FF 7F800000
	v_cmp_ne_u32_e64 s[10:11], s58, v10                        // 00000003B9D8: D0CD000A 0002143A
	s_and_saveexec_b64 s[18:19], s[10:11]                      // 00000003B9E0: BE92200A
	s_xor_b64 s[10:11], exec, s[18:19]                         // 00000003B9E4: 888A127E
	v_bfe_u32 v10, v9, 16, 1                                   // 00000003B9E8: D1C8000A 02052109
	v_add3_u32 v10, v9, v10, s59                               // 00000003B9F0: D1FF000A 00EE1509
	s_andn2_saveexec_b64 s[18:19], s[10:11]                    // 00000003B9F8: BE92230A
	s_cbranch_execz 65452                                      // 00000003B9FC: BF88FFAC <EpCombineIntraNodeKernel_bf16_nop2p+0x28b0>
	v_or_b32_e32 v10, 0x10000, v9                              // 00000003BA00: 281412FF 00010000
	v_cmp_eq_u32_sdwa s[10:11], v9, v21 src0_sel:WORD_0 src1_sel:DWORD// 00000003BA08: 7D942AF9 06048A09
	s_nop 1                                                    // 00000003BA10: BF800001
	v_cndmask_b32_e64 v10, v10, v9, s[10:11]                   // 00000003BA14: D100000A 002A130A
	s_branch 65444                                             // 00000003BA1C: BF82FFA4 <EpCombineIntraNodeKernel_bf16_nop2p+0x28b0>
	s_or_b64 exec, exec, s[14:15]                              // 00000003BA20: 87FE0E7E
	s_or_b64 exec, exec, s[12:13]                              // 00000003BA24: 87FE0C7E
	v_lshl_add_u64 v[0:1], v[8:9], 0, v[42:43]                 // 00000003BA28: D2080000 04A90108
	v_cmp_lt_u64_e32 vcc, v[0:1], v[36:37]                     // 00000003BA30: 7DD24900
	s_and_saveexec_b64 s[8:9], vcc                             // 00000003BA34: BE88206A
	s_cbranch_execz 67                                         // 00000003BA38: BF880043 <EpCombineIntraNodeKernel_bf16_nop2p+0x2b48>
	ds_read2_b64 v[4:7], v27 offset1:1                         // 00000003BA3C: D8EE0100 0400001B
	v_lshlrev_b64 v[8:9], 1, v[0:1]                            // 00000003BA44: D28F0008 00020081
	s_mov_b64 s[10:11], 0                                      // 00000003BA4C: BE8A0180
	s_waitcnt lgkmcnt(0)                                       // 00000003BA50: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[4:5]                            // 00000003BA54: 7DDA0880
	v_cmp_ne_u64_e64 s[4:5], 0, v[6:7]                         // 00000003BA58: D0ED0004 00020C80
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[8:9]                   // 00000003BA60: D2080002 04210106
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[8:9]                   // 00000003BA68: D2080004 04210104
	s_branch 16                                                // 00000003BA70: BF820010 <EpCombineIntraNodeKernel_bf16_nop2p+0x2ab4>
	s_or_b64 exec, exec, s[12:13]                              // 00000003BA74: 87FE0C7E
	v_lshl_add_u64 v[8:9], v[0:1], 1, v[34:35]                 // 00000003BA78: D2080008 04890300
	v_lshl_add_u64 v[0:1], v[0:1], 0, 64                       // 00000003BA80: D2080000 03010100
	v_cmp_ge_u64_e64 s[6:7], v[0:1], v[36:37]                  // 00000003BA88: D0EE0006 00024900
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[86:87]                 // 00000003BA90: D2080002 01590102
	s_or_b64 s[10:11], s[6:7], s[10:11]                        // 00000003BA98: 878A0A06
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[86:87]                 // 00000003BA9C: D2080004 01590104
	flat_store_short_d16_hi v[8:9], v7                         // 00000003BAA4: DC6C0000 00000708
	s_andn2_b64 exec, exec, s[10:11]                           // 00000003BAAC: 89FE0A7E
	s_cbranch_execz 37                                         // 00000003BAB0: BF880025 <EpCombineIntraNodeKernel_bf16_nop2p+0x2b48>
	v_mov_b32_e32 v6, 0                                        // 00000003BAB4: 7E0C0280
	s_and_saveexec_b64 s[6:7], vcc                             // 00000003BAB8: BE86206A
	s_cbranch_execz 5                                          // 00000003BABC: BF880005 <EpCombineIntraNodeKernel_bf16_nop2p+0x2ad4>
	flat_load_ushort v6, v[4:5]                                // 00000003BAC0: DC480000 06000004
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003BAC8: BF8C0070
	v_lshlrev_b32_e32 v6, 16, v6                               // 00000003BACC: 240C0C90
	v_add_f32_e32 v6, 0, v6                                    // 00000003BAD0: 020C0C80
	s_or_b64 exec, exec, s[6:7]                                // 00000003BAD4: 87FE067E
	s_and_saveexec_b64 s[6:7], s[4:5]                          // 00000003BAD8: BE862004
	s_cbranch_execz 5                                          // 00000003BADC: BF880005 <EpCombineIntraNodeKernel_bf16_nop2p+0x2af4>
	flat_load_ushort v7, v[2:3]                                // 00000003BAE0: DC480000 07000002
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003BAE8: BF8C0070
	v_lshlrev_b32_e32 v7, 16, v7                               // 00000003BAEC: 240E0E90
	v_add_f32_e32 v6, v6, v7                                   // 00000003BAF0: 020C0F06
	s_or_b64 exec, exec, s[6:7]                                // 00000003BAF4: 87FE067E
	v_and_b32_e32 v7, 0x7f800000, v6                           // 00000003BAF8: 260E0CFF 7F800000
	v_cmp_ne_u32_e64 s[6:7], s58, v7                           // 00000003BB00: D0CD0006 00020E3A
	s_and_saveexec_b64 s[12:13], s[6:7]                        // 00000003BB08: BE8C2006
	s_xor_b64 s[6:7], exec, s[12:13]                           // 00000003BB0C: 88860C7E
	v_bfe_u32 v7, v6, 16, 1                                    // 00000003BB10: D1C80007 02052106
	v_add3_u32 v7, v6, v7, s59                                 // 00000003BB18: D1FF0007 00EE0F06
	s_andn2_saveexec_b64 s[12:13], s[6:7]                      // 00000003BB20: BE8C2306
	s_cbranch_execz 65491                                      // 00000003BB24: BF88FFD3 <EpCombineIntraNodeKernel_bf16_nop2p+0x2a74>
	v_or_b32_e32 v7, 0x10000, v6                               // 00000003BB28: 280E0CFF 00010000
	v_cmp_eq_u32_sdwa s[6:7], v6, v21 src0_sel:WORD_0 src1_sel:DWORD// 00000003BB30: 7D942AF9 06048606
	s_nop 1                                                    // 00000003BB38: BF800001
	v_cndmask_b32_e64 v7, v7, v6, s[6:7]                       // 00000003BB3C: D1000007 001A0D07
	s_branch 65483                                             // 00000003BB44: BF82FFCB <EpCombineIntraNodeKernel_bf16_nop2p+0x2a74>
	s_or_b64 exec, exec, s[8:9]                                // 00000003BB48: 87FE087E
	s_mov_b64 s[4:5], 0                                        // 00000003BB4C: BE840180
	s_mov_b64 s[6:7], 0                                        // 00000003BB50: BE860180
	s_mov_b64 s[92:93], 0                                      // 00000003BB54: BEDC0180
	s_and_b64 vcc, exec, s[6:7]                                // 00000003BB58: 86EA067E
	s_cbranch_vccz 3                                           // 00000003BB5C: BF860003 <EpCombineIntraNodeKernel_bf16_nop2p+0x2b6c>
	s_cmp_lg_u32 s90, 1                                        // 00000003BB60: BF07815A
	s_mov_b64 s[92:93], -1                                     // 00000003BB64: BEDC01C1
	s_cselect_b64 s[4:5], -1, 0                                // 00000003BB68: 858480C1
	s_and_b64 vcc, exec, s[4:5]                                // 00000003BB6C: 86EA047E
	v_cmp_lt_u64_e64 s[4:5], s[84:85], v[36:37]                // 00000003BB70: D0E90004 00024854
	s_cbranch_vccz 274                                         // 00000003BB78: BF860112 <EpCombineIntraNodeKernel_bf16_nop2p+0x2fc4>
	s_ashr_i32 s91, s90, 31                                    // 00000003BB7C: 905B9F5A
	v_mov_b64_e32 v[4:5], 0                                    // 00000003BB80: 7E087080
	s_and_saveexec_b64 s[6:7], s[4:5]                          // 00000003BB84: BE862004
	s_cbranch_execz 204                                        // 00000003BB88: BF8800CC <EpCombineIntraNodeKernel_bf16_nop2p+0x2ebc>
	s_cmp_lg_u32 s90, 0                                        // 00000003BB8C: BF07805A
	s_mov_b32 s26, s66                                         // 00000003BB90: BE9A0042
	s_cselect_b64 s[4:5], -1, 0                                // 00000003BB94: 858480C1
	s_and_b32 s66, s90, 3                                      // 00000003BB98: 8642835A
	s_cmp_gt_u32 s90, 3                                        // 00000003BB9C: BF08835A
	s_cselect_b64 s[8:9], -1, 0                                // 00000003BBA0: 858880C1
	s_and_b32 s10, s90, -4                                     // 00000003BBA4: 860AC45A
	v_lshlrev_b32_e32 v20, 1, v28                              // 00000003BBA8: 24283881
	s_cmp_lg_u32 s66, 0                                        // 00000003BBAC: BF078042
	v_lshrrev_b64 v[0:1], 7, v[36:37]                          // 00000003BBB0: D2900000 00024887
	v_lshl_add_u64 v[2:3], v[34:35], 0, v[20:21]               // 00000003BBB8: D2080002 04510122
	s_mov_b32 s11, s91                                         // 00000003BBC0: BE8B005B
	s_mov_b64 s[12:13], 0                                      // 00000003BBC4: BE8C0180
	s_cselect_b64 s[14:15], -1, 0                              // 00000003BBC8: 858E80C1
	s_mov_b64 s[16:17], 0                                      // 00000003BBCC: BE900180
	s_mov_b64 s[18:19], 0                                      // 00000003BBD0: BE920180
	s_branch 18                                                // 00000003BBD4: BF820012 <EpCombineIntraNodeKernel_bf16_nop2p+0x2c20>
	s_or_b64 exec, exec, s[20:21]                              // 00000003BBD8: 87FE147E
	v_lshrrev_b32_e32 v5, 16, v6                               // 00000003BBDC: 200A0C90
	v_lshl_add_u64 v[6:7], s[16:17], 1, v[2:3]                 // 00000003BBE0: D2080006 04090210
	s_add_u32 s16, s16, 0x80                                   // 00000003BBE8: 8010FF10 00000080
	s_addc_u32 s17, s17, 0                                     // 00000003BBF0: 82118011
	s_add_u32 s18, s18, 1                                      // 00000003BBF4: 80128112
	s_addc_u32 s19, s19, 0                                     // 00000003BBF8: 82138013
	v_and_or_b32 v4, v4, s82, v5                               // 00000003BBFC: D2010004 0414A504
	v_cmp_eq_u64_e32 vcc, s[18:19], v[0:1]                     // 00000003BC04: 7DD40012
	flat_store_dword v[6:7], v4 nt                             // 00000003BC08: DC720000 00000406
	s_or_b64 s[12:13], vcc, s[12:13]                           // 00000003BC10: 878C0C6A
	v_mov_b64_e32 v[4:5], s[16:17]                             // 00000003BC14: 7E087010
	s_andn2_b64 exec, exec, s[12:13]                           // 00000003BC18: 89FE0C7E
	s_cbranch_execz 165                                        // 00000003BC1C: BF8800A5 <EpCombineIntraNodeKernel_bf16_nop2p+0x2eb4>
	s_andn2_b64 vcc, exec, s[4:5]                              // 00000003BC20: 89EA047E
	s_cbranch_vccnz 89                                         // 00000003BC24: BF870059 <EpCombineIntraNodeKernel_bf16_nop2p+0x2d8c>
	v_mov_b32_e32 v4, v21                                      // 00000003BC28: 7E080315
	v_mov_b32_e32 v5, v21                                      // 00000003BC2C: 7E0A0315
	s_andn2_b64 vcc, exec, s[8:9]                              // 00000003BC30: 89EA087E
	s_cbranch_vccnz 88                                         // 00000003BC34: BF870058 <EpCombineIntraNodeKernel_bf16_nop2p+0x2d98>
	v_mov_b32_e32 v8, v27                                      // 00000003BC38: 7E10031B
	s_mov_b64 s[20:21], s[10:11]                               // 00000003BC3C: BE94010A
	s_branch 6                                                 // 00000003BC40: BF820006 <EpCombineIntraNodeKernel_bf16_nop2p+0x2c5c>
	s_or_b64 exec, exec, s[22:23]                              // 00000003BC44: 87FE167E
	s_add_u32 s20, s20, -4                                     // 00000003BC48: 8014C414
	s_addc_u32 s21, s21, -1                                    // 00000003BC4C: 8215C115
	s_cmp_eq_u64 s[20:21], 0                                   // 00000003BC50: BF128014
	v_add_u32_e32 v8, 32, v8                                   // 00000003BC54: 681010A0
	s_cbranch_scc1 83                                          // 00000003BC58: BF850053 <EpCombineIntraNodeKernel_bf16_nop2p+0x2da8>
	ds_read_b64 v[6:7], v8                                     // 00000003BC5C: D8EC0000 06000008
	s_waitcnt lgkmcnt(0)                                       // 00000003BC64: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[6:7]                            // 00000003BC68: 7DDA0C80
	s_and_saveexec_b64 s[22:23], vcc                           // 00000003BC6C: BE96206A
	s_cbranch_execz 12                                         // 00000003BC70: BF88000C <EpCombineIntraNodeKernel_bf16_nop2p+0x2ca4>
	v_lshl_add_u64 v[6:7], s[16:17], 1, v[6:7]                 // 00000003BC74: D2080006 04190210
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[20:21]                 // 00000003BC7C: D2080006 04510106
	flat_load_dword v6, v[6:7] nt                              // 00000003BC84: DC520000 06000006
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003BC8C: BF8C0070
	v_and_b32_e32 v7, 0xffff0000, v6                           // 00000003BC90: 260E0CFF FFFF0000
	v_lshlrev_b32_e32 v6, 16, v6                               // 00000003BC98: 240C0C90
	v_pk_add_f32 v[4:5], v[4:5], v[6:7]                        // 00000003BC9C: D3B24004 18020D04
	s_or_b64 exec, exec, s[22:23]                              // 00000003BCA4: 87FE167E
	ds_read_b64 v[6:7], v8 offset:8                            // 00000003BCA8: D8EC0008 06000008
	s_waitcnt lgkmcnt(0)                                       // 00000003BCB0: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[6:7]                            // 00000003BCB4: 7DDA0C80
	s_and_saveexec_b64 s[22:23], vcc                           // 00000003BCB8: BE96206A
	s_cbranch_execz 12                                         // 00000003BCBC: BF88000C <EpCombineIntraNodeKernel_bf16_nop2p+0x2cf0>
	v_lshl_add_u64 v[6:7], s[16:17], 1, v[6:7]                 // 00000003BCC0: D2080006 04190210
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[20:21]                 // 00000003BCC8: D2080006 04510106
	flat_load_dword v6, v[6:7] nt                              // 00000003BCD0: DC520000 06000006
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003BCD8: BF8C0070
	v_and_b32_e32 v7, 0xffff0000, v6                           // 00000003BCDC: 260E0CFF FFFF0000
	v_lshlrev_b32_e32 v6, 16, v6                               // 00000003BCE4: 240C0C90
	v_pk_add_f32 v[4:5], v[4:5], v[6:7]                        // 00000003BCE8: D3B24004 18020D04
	s_or_b64 exec, exec, s[22:23]                              // 00000003BCF0: 87FE167E
	ds_read_b64 v[6:7], v8 offset:16                           // 00000003BCF4: D8EC0010 06000008
	s_waitcnt lgkmcnt(0)                                       // 00000003BCFC: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[6:7]                            // 00000003BD00: 7DDA0C80
	s_and_saveexec_b64 s[22:23], vcc                           // 00000003BD04: BE96206A
	s_cbranch_execz 12                                         // 00000003BD08: BF88000C <EpCombineIntraNodeKernel_bf16_nop2p+0x2d3c>
	v_lshl_add_u64 v[6:7], s[16:17], 1, v[6:7]                 // 00000003BD0C: D2080006 04190210
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[20:21]                 // 00000003BD14: D2080006 04510106
	flat_load_dword v6, v[6:7] nt                              // 00000003BD1C: DC520000 06000006
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003BD24: BF8C0070
	v_and_b32_e32 v7, 0xffff0000, v6                           // 00000003BD28: 260E0CFF FFFF0000
	v_lshlrev_b32_e32 v6, 16, v6                               // 00000003BD30: 240C0C90
	v_pk_add_f32 v[4:5], v[4:5], v[6:7]                        // 00000003BD34: D3B24004 18020D04
	s_or_b64 exec, exec, s[22:23]                              // 00000003BD3C: 87FE167E
	ds_read_b64 v[6:7], v8 offset:24                           // 00000003BD40: D8EC0018 06000008
	s_waitcnt lgkmcnt(0)                                       // 00000003BD48: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[6:7]                            // 00000003BD4C: 7DDA0C80
	s_and_saveexec_b64 s[22:23], vcc                           // 00000003BD50: BE96206A
	s_cbranch_execz 65467                                      // 00000003BD54: BF88FFBB <EpCombineIntraNodeKernel_bf16_nop2p+0x2c44>
	v_lshl_add_u64 v[6:7], s[16:17], 1, v[6:7]                 // 00000003BD58: D2080006 04190210
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[20:21]                 // 00000003BD60: D2080006 04510106
	flat_load_dword v6, v[6:7] nt                              // 00000003BD68: DC520000 06000006
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003BD70: BF8C0070
	v_and_b32_e32 v7, 0xffff0000, v6                           // 00000003BD74: 260E0CFF FFFF0000
	v_lshlrev_b32_e32 v6, 16, v6                               // 00000003BD7C: 240C0C90
	v_pk_add_f32 v[4:5], v[4:5], v[6:7]                        // 00000003BD80: D3B24004 18020D04
	s_branch 65454                                             // 00000003BD88: BF82FFAE <EpCombineIntraNodeKernel_bf16_nop2p+0x2c44>
	v_mov_b32_e32 v5, 0                                        // 00000003BD8C: 7E0A0280
	v_mov_b32_e32 v4, 0                                        // 00000003BD90: 7E080280
	s_branch 36                                                // 00000003BD94: BF820024 <EpCombineIntraNodeKernel_bf16_nop2p+0x2e28>
	s_mov_b64 s[20:21], 0                                      // 00000003BD98: BE940180
	s_andn2_b64 vcc, exec, s[14:15]                            // 00000003BD9C: 89EA0E7E
	s_cbranch_vccz 4                                           // 00000003BDA0: BF860004 <EpCombineIntraNodeKernel_bf16_nop2p+0x2db4>
	s_branch 32                                                // 00000003BDA4: BF820020 <EpCombineIntraNodeKernel_bf16_nop2p+0x2e28>
	s_mov_b64 s[20:21], s[10:11]                               // 00000003BDA8: BE94010A
	s_andn2_b64 vcc, exec, s[14:15]                            // 00000003BDAC: 89EA0E7E
	s_cbranch_vccnz 29                                         // 00000003BDB0: BF87001D <EpCombineIntraNodeKernel_bf16_nop2p+0x2e28>
	v_lshl_add_u32 v8, s20, 3, v27                             // 00000003BDB4: D1FD0008 046D0614
	s_mov_b64 s[20:21], s[66:67]                               // 00000003BDBC: BE940142
	s_branch 6                                                 // 00000003BDC0: BF820006 <EpCombineIntraNodeKernel_bf16_nop2p+0x2ddc>
	s_or_b64 exec, exec, s[22:23]                              // 00000003BDC4: 87FE167E
	s_add_u32 s20, s20, -1                                     // 00000003BDC8: 8014C114
	s_addc_u32 s21, s21, -1                                    // 00000003BDCC: 8215C115
	s_cmp_lg_u64 s[20:21], 0                                   // 00000003BDD0: BF138014
	v_add_u32_e32 v8, 8, v8                                    // 00000003BDD4: 68101088
	s_cbranch_scc0 19                                          // 00000003BDD8: BF840013 <EpCombineIntraNodeKernel_bf16_nop2p+0x2e28>
	ds_read_b64 v[6:7], v8                                     // 00000003BDDC: D8EC0000 06000008
	s_waitcnt lgkmcnt(0)                                       // 00000003BDE4: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[6:7]                            // 00000003BDE8: 7DDA0C80
	s_and_saveexec_b64 s[22:23], vcc                           // 00000003BDEC: BE96206A
	s_cbranch_execz 65524                                      // 00000003BDF0: BF88FFF4 <EpCombineIntraNodeKernel_bf16_nop2p+0x2dc4>
	v_lshl_add_u64 v[6:7], s[16:17], 1, v[6:7]                 // 00000003BDF4: D2080006 04190210
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[20:21]                 // 00000003BDFC: D2080006 04510106
	flat_load_dword v6, v[6:7] nt                              // 00000003BE04: DC520000 06000006
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003BE0C: BF8C0070
	v_and_b32_e32 v7, 0xffff0000, v6                           // 00000003BE10: 260E0CFF FFFF0000
	v_lshlrev_b32_e32 v6, 16, v6                               // 00000003BE18: 240C0C90
	v_pk_add_f32 v[4:5], v[4:5], v[6:7]                        // 00000003BE1C: D3B24004 18020D04
	s_branch 65511                                             // 00000003BE24: BF82FFE7 <EpCombineIntraNodeKernel_bf16_nop2p+0x2dc4>
	v_and_b32_e32 v6, 0x7f800000, v4                           // 00000003BE28: 260C08FF 7F800000
	v_cmp_ne_u32_e32 vcc, s58, v6                              // 00000003BE30: 7D9A0C3A
	s_and_saveexec_b64 s[20:21], vcc                           // 00000003BE34: BE94206A
	s_xor_b64 s[20:21], exec, s[20:21]                         // 00000003BE38: 8894147E
	v_bfe_u32 v6, v4, 16, 1                                    // 00000003BE3C: D1C80006 02052104
	v_add3_u32 v6, v4, v6, s59                                 // 00000003BE44: D1FF0006 00EE0D04
	s_andn2_saveexec_b64 s[20:21], s[20:21]                    // 00000003BE4C: BE942314
	v_or_b32_e32 v6, 0x10000, v4                               // 00000003BE50: 280C08FF 00010000
	v_cmp_eq_u32_sdwa vcc, v4, v21 src0_sel:WORD_0 src1_sel:DWORD// 00000003BE58: 7D942AF9 06040004
	s_nop 1                                                    // 00000003BE60: BF800001
	v_cndmask_b32_e32 v6, v6, v4, vcc                          // 00000003BE64: 000C0906
	s_or_b64 exec, exec, s[20:21]                              // 00000003BE68: 87FE147E
	v_and_b32_e32 v4, 0x7f800000, v5                           // 00000003BE6C: 26080AFF 7F800000
	v_cmp_ne_u32_e32 vcc, s58, v4                              // 00000003BE74: 7D9A083A
	s_and_saveexec_b64 s[20:21], vcc                           // 00000003BE78: BE94206A
	s_xor_b64 s[20:21], exec, s[20:21]                         // 00000003BE7C: 8894147E
	v_bfe_u32 v4, v5, 16, 1                                    // 00000003BE80: D1C80004 02052105
	v_add3_u32 v4, v5, v4, s59                                 // 00000003BE88: D1FF0004 00EE0905
	s_andn2_saveexec_b64 s[20:21], s[20:21]                    // 00000003BE90: BE942314
	s_cbranch_execz 65360                                      // 00000003BE94: BF88FF50 <EpCombineIntraNodeKernel_bf16_nop2p+0x2bd8>
	v_or_b32_e32 v4, 0x10000, v5                               // 00000003BE98: 28080AFF 00010000
	v_cmp_eq_u32_sdwa vcc, v5, v21 src0_sel:WORD_0 src1_sel:DWORD// 00000003BEA0: 7D942AF9 06040005
	s_nop 1                                                    // 00000003BEA8: BF800001
	v_cndmask_b32_e32 v4, v4, v5, vcc                          // 00000003BEAC: 00080B04
	s_branch 65353                                             // 00000003BEB0: BF82FF49 <EpCombineIntraNodeKernel_bf16_nop2p+0x2bd8>
	s_or_b64 exec, exec, s[12:13]                              // 00000003BEB4: 87FE0C7E
	s_mov_b32 s66, s26                                         // 00000003BEB8: BEC2001A
	s_or_b64 exec, exec, s[6:7]                                // 00000003BEBC: 87FE067E
	v_or_b32_e32 v4, v4, v42                                   // 00000003BEC0: 28085504
	v_cmp_lt_u64_e32 vcc, v[4:5], v[36:37]                     // 00000003BEC4: 7DD24904
	s_and_saveexec_b64 s[4:5], vcc                             // 00000003BEC8: BE84206A
	s_cbranch_execz 59                                         // 00000003BECC: BF88003B <EpCombineIntraNodeKernel_bf16_nop2p+0x2fbc>
	s_cmp_lg_u32 s90, 0                                        // 00000003BED0: BF07805A
	s_mov_b64 s[6:7], 0                                        // 00000003BED4: BE860180
	s_cselect_b64 s[8:9], -1, 0                                // 00000003BED8: 858880C1
	s_branch 11                                                // 00000003BEDC: BF82000B <EpCombineIntraNodeKernel_bf16_nop2p+0x2f0c>
	s_or_b64 exec, exec, s[10:11]                              // 00000003BEE0: 87FE0A7E
	v_lshl_add_u64 v[2:3], v[4:5], 1, v[34:35]                 // 00000003BEE4: D2080002 04890304
	v_lshl_add_u64 v[4:5], v[4:5], 0, 64                       // 00000003BEEC: D2080004 03010104
	v_cmp_ge_u64_e32 vcc, v[4:5], v[36:37]                     // 00000003BEF4: 7DDC4904
	s_or_b64 s[6:7], vcc, s[6:7]                               // 00000003BEF8: 8786066A
	flat_store_short_d16_hi v[2:3], v0                         // 00000003BEFC: DC6C0000 00000002
	s_andn2_b64 exec, exec, s[6:7]                             // 00000003BF04: 89FE067E
	s_cbranch_execz 44                                         // 00000003BF08: BF88002C <EpCombineIntraNodeKernel_bf16_nop2p+0x2fbc>
	v_mov_b32_e32 v2, 0                                        // 00000003BF0C: 7E040280
	s_andn2_b64 vcc, exec, s[8:9]                              // 00000003BF10: 89EA087E
	s_cbranch_vccnz 23                                         // 00000003BF14: BF870017 <EpCombineIntraNodeKernel_bf16_nop2p+0x2f74>
	v_mov_b32_e32 v3, v27                                      // 00000003BF18: 7E06031B
	s_mov_b64 s[10:11], s[90:91]                               // 00000003BF1C: BE8A015A
	s_branch 6                                                 // 00000003BF20: BF820006 <EpCombineIntraNodeKernel_bf16_nop2p+0x2f3c>
	s_or_b64 exec, exec, s[12:13]                              // 00000003BF24: 87FE0C7E
	s_add_u32 s10, s10, -1                                     // 00000003BF28: 800AC10A
	s_addc_u32 s11, s11, -1                                    // 00000003BF2C: 820BC10B
	s_cmp_eq_u64 s[10:11], 0                                   // 00000003BF30: BF12800A
	v_add_u32_e32 v3, 8, v3                                    // 00000003BF34: 68060688
	s_cbranch_scc1 14                                          // 00000003BF38: BF85000E <EpCombineIntraNodeKernel_bf16_nop2p+0x2f74>
	ds_read_b64 v[0:1], v3                                     // 00000003BF3C: D8EC0000 00000003
	s_waitcnt lgkmcnt(0)                                       // 00000003BF44: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[0:1]                            // 00000003BF48: 7DDA0080
	s_and_saveexec_b64 s[12:13], vcc                           // 00000003BF4C: BE8C206A
	s_cbranch_execz 65524                                      // 00000003BF50: BF88FFF4 <EpCombineIntraNodeKernel_bf16_nop2p+0x2f24>
	v_lshl_add_u64 v[0:1], v[4:5], 1, v[0:1]                   // 00000003BF54: D2080000 04010304
	flat_load_ushort v0, v[0:1]                                // 00000003BF5C: DC480000 00000000
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003BF64: BF8C0070
	v_lshlrev_b32_e32 v0, 16, v0                               // 00000003BF68: 24000090
	v_add_f32_e32 v2, v2, v0                                   // 00000003BF6C: 02040102
	s_branch 65516                                             // 00000003BF70: BF82FFEC <EpCombineIntraNodeKernel_bf16_nop2p+0x2f24>
	v_and_b32_e32 v0, 0x7f800000, v2                           // 00000003BF74: 260004FF 7F800000
	v_cmp_ne_u32_e32 vcc, s58, v0                              // 00000003BF7C: 7D9A003A
	s_and_saveexec_b64 s[10:11], vcc                           // 00000003BF80: BE8A206A
	s_xor_b64 s[10:11], exec, s[10:11]                         // 00000003BF84: 888A0A7E
	v_bfe_u32 v0, v2, 16, 1                                    // 00000003BF88: D1C80000 02052102
	v_add3_u32 v0, v2, v0, s59                                 // 00000003BF90: D1FF0000 00EE0102
	s_andn2_saveexec_b64 s[10:11], s[10:11]                    // 00000003BF98: BE8A230A
	s_cbranch_execz 65488                                      // 00000003BF9C: BF88FFD0 <EpCombineIntraNodeKernel_bf16_nop2p+0x2ee0>
	v_or_b32_e32 v0, 0x10000, v2                               // 00000003BFA0: 280004FF 00010000
	v_cmp_eq_u32_sdwa vcc, v2, v21 src0_sel:WORD_0 src1_sel:DWORD// 00000003BFA8: 7D942AF9 06040002
	s_nop 1                                                    // 00000003BFB0: BF800001
	v_cndmask_b32_e32 v0, v0, v2, vcc                          // 00000003BFB4: 00000500
	s_branch 65481                                             // 00000003BFB8: BF82FFC9 <EpCombineIntraNodeKernel_bf16_nop2p+0x2ee0>
	s_or_b64 exec, exec, s[4:5]                                // 00000003BFBC: 87FE047E
	s_mov_b64 s[92:93], 0                                      // 00000003BFC0: BEDC0180
	s_and_b64 vcc, exec, s[92:93]                              // 00000003BFC4: 86EA5C7E
	s_cbranch_vccz 161                                         // 00000003BFC8: BF8600A1 <EpCombineIntraNodeKernel_bf16_nop2p+0x3250>
	v_cmp_lt_u64_e32 vcc, s[84:85], v[36:37]                   // 00000003BFCC: 7DD24854
	v_mov_b64_e32 v[8:9], 0                                    // 00000003BFD0: 7E107080
	s_and_saveexec_b64 s[8:9], vcc                             // 00000003BFD4: BE88206A
	s_cbranch_execz 99                                         // 00000003BFD8: BF880063 <EpCombineIntraNodeKernel_bf16_nop2p+0x3168>
	ds_read_b64 v[6:7], v27                                    // 00000003BFDC: D8EC0000 0600001B
	v_lshlrev_b32_e32 v20, 1, v26                              // 00000003BFE4: 24283481
	v_lshrrev_b64 v[0:1], 7, v[36:37]                          // 00000003BFE8: D2900000 00024887
	s_mov_b64 s[10:11], 0                                      // 00000003BFF0: BE8A0180
	v_lshl_add_u64 v[2:3], v[34:35], 0, v[20:21]               // 00000003BFF4: D2080002 04510122
	s_waitcnt lgkmcnt(0)                                       // 00000003BFFC: BF8CC07F
	v_lshl_add_u64 v[4:5], v[6:7], 0, v[20:21]                 // 00000003C000: D2080004 04510106
	v_cmp_eq_u64_e32 vcc, 0, v[6:7]                            // 00000003C008: 7DD40C80
	v_cmp_ne_u64_e64 s[4:5], 0, v[6:7]                         // 00000003C00C: D0ED0004 00020C80
	v_mov_b32_e32 v10, 0                                       // 00000003C014: 7E140280
	v_mov_b32_e32 v7, 0                                        // 00000003C018: 7E0E0280
	s_mov_b64 s[12:13], 0                                      // 00000003C01C: BE8C0180
	s_branch 21                                                // 00000003C020: BF820015 <EpCombineIntraNodeKernel_bf16_nop2p+0x3078>
	s_or_b64 exec, exec, s[14:15]                              // 00000003C024: 87FE0E7E
	v_lshl_add_u64 v[12:13], s[12:13], 1, v[2:3]               // 00000003C028: D208000C 0409020C
	s_add_u32 s12, s12, 0x80                                   // 00000003C030: 800CFF0C 00000080
	v_lshl_add_u64 v[0:1], v[0:1], 0, -1                       // 00000003C038: D2080000 03050100
	v_lshrrev_b32_e32 v6, 16, v6                               // 00000003C040: 200C0C90
	s_addc_u32 s13, s13, 0                                     // 00000003C044: 820D800D
	v_cmp_eq_u64_e64 s[6:7], 0, v[0:1]                         // 00000003C048: D0EA0006 00020080
	v_and_or_b32 v6, v9, s82, v6                               // 00000003C050: D2010006 0418A509
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[88:89]                 // 00000003C058: D2080004 01610104
	s_or_b64 s[10:11], s[6:7], s[10:11]                        // 00000003C060: 878A0A06
	v_mov_b64_e32 v[8:9], s[12:13]                             // 00000003C064: 7E10700C
	flat_store_dword v[12:13], v6 nt                           // 00000003C068: DC720000 0000060C
	s_andn2_b64 exec, exec, s[10:11]                           // 00000003C070: 89FE0A7E
	s_cbranch_execz 59                                         // 00000003C074: BF88003B <EpCombineIntraNodeKernel_bf16_nop2p+0x3164>
	s_mov_b32 s6, 0xffff                                       // 00000003C078: BE8600FF 0000FFFF
	v_and_or_b32 v10, v10, s6, v7                              // 00000003C080: D201000A 041C0D0A
	s_and_saveexec_b64 s[6:7], s[4:5]                          // 00000003C088: BE862004
	s_cbranch_execz 2                                          // 00000003C08C: BF880002 <EpCombineIntraNodeKernel_bf16_nop2p+0x3098>
	flat_load_dword v10, v[4:5] nt                             // 00000003C090: DC520000 0A000004
	s_or_b64 exec, exec, s[6:7]                                // 00000003C098: 87FE067E
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003C09C: BF8C0070
	v_lshlrev_b32_e32 v6, 16, v10                              // 00000003C0A0: 240C1490
	v_and_b32_e32 v7, 0xffff0000, v10                          // 00000003C0A4: 260E14FF FFFF0000
	v_pk_add_f32 v[8:9], v[6:7], 0 op_sel_hi:[1,0]             // 00000003C0AC: D3B24008 08010106
	s_nop 0                                                    // 00000003C0B4: BF800000
	v_cndmask_b32_e64 v8, v8, 0, vcc                           // 00000003C0B8: D1000008 01A90108
	v_and_b32_e32 v6, 0x7f800000, v8                           // 00000003C0C0: 260C10FF 7F800000
	v_cmp_ne_u32_e64 s[6:7], s58, v6                           // 00000003C0C8: D0CD0006 00020C3A
	s_and_saveexec_b64 s[14:15], s[6:7]                        // 00000003C0D0: BE8E2006
	s_xor_b64 s[6:7], exec, s[14:15]                           // 00000003C0D4: 88860E7E
	v_bfe_u32 v6, v8, 16, 1                                    // 00000003C0D8: D1C80006 02052108
	v_add3_u32 v6, v8, v6, s59                                 // 00000003C0E0: D1FF0006 00EE0D08
	s_andn2_saveexec_b64 s[14:15], s[6:7]                      // 00000003C0E8: BE8E2306
	v_or_b32_e32 v6, 0x10000, v8                               // 00000003C0EC: 280C10FF 00010000
	v_cmp_eq_u32_sdwa s[6:7], v8, v21 src0_sel:WORD_0 src1_sel:DWORD// 00000003C0F4: 7D942AF9 06048608
	s_nop 1                                                    // 00000003C0FC: BF800001
	v_cndmask_b32_e64 v6, v6, v8, s[6:7]                       // 00000003C100: D1000006 001A1106
	s_or_b64 exec, exec, s[14:15]                              // 00000003C108: 87FE0E7E
	v_cndmask_b32_e64 v8, v9, 0, vcc                           // 00000003C10C: D1000008 01A90109
	v_and_b32_e32 v9, 0x7f800000, v8                           // 00000003C114: 261210FF 7F800000
	v_cmp_ne_u32_e64 s[6:7], s58, v9                           // 00000003C11C: D0CD0006 0002123A
	s_and_saveexec_b64 s[14:15], s[6:7]                        // 00000003C124: BE8E2006
	s_xor_b64 s[6:7], exec, s[14:15]                           // 00000003C128: 88860E7E
	v_bfe_u32 v9, v8, 16, 1                                    // 00000003C12C: D1C80009 02052108
	v_add3_u32 v9, v8, v9, s59                                 // 00000003C134: D1FF0009 00EE1308
	s_andn2_saveexec_b64 s[14:15], s[6:7]                      // 00000003C13C: BE8E2306
	s_cbranch_execz 65464                                      // 00000003C140: BF88FFB8 <EpCombineIntraNodeKernel_bf16_nop2p+0x3024>
	v_or_b32_e32 v9, 0x10000, v8                               // 00000003C144: 281210FF 00010000
	v_cmp_eq_u32_sdwa s[6:7], v8, v21 src0_sel:WORD_0 src1_sel:DWORD// 00000003C14C: 7D942AF9 06048608
	s_nop 1                                                    // 00000003C154: BF800001
	v_cndmask_b32_e64 v9, v9, v8, s[6:7]                       // 00000003C158: D1000009 001A1109
	s_branch 65456                                             // 00000003C160: BF82FFB0 <EpCombineIntraNodeKernel_bf16_nop2p+0x3024>
	s_or_b64 exec, exec, s[10:11]                              // 00000003C164: 87FE0A7E
	s_or_b64 exec, exec, s[8:9]                                // 00000003C168: 87FE087E
	v_lshl_add_u64 v[0:1], v[8:9], 0, v[42:43]                 // 00000003C16C: D2080000 04A90108
	v_cmp_lt_u64_e32 vcc, v[0:1], v[36:37]                     // 00000003C174: 7DD24900
	s_and_saveexec_b64 s[6:7], vcc                             // 00000003C178: BE86206A
	s_cbranch_execz 51                                         // 00000003C17C: BF880033 <EpCombineIntraNodeKernel_bf16_nop2p+0x324c>
	ds_read_b64 v[2:3], v27                                    // 00000003C180: D8EC0000 0200001B
	s_mov_b64 s[8:9], 0                                        // 00000003C188: BE880180
	s_waitcnt lgkmcnt(0)                                       // 00000003C18C: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[2:3]                            // 00000003C190: 7DDA0480
	v_lshl_add_u64 v[2:3], v[0:1], 1, v[2:3]                   // 00000003C194: D2080002 04090300
	s_branch 14                                                // 00000003C19C: BF82000E <EpCombineIntraNodeKernel_bf16_nop2p+0x31d8>
	s_or_b64 exec, exec, s[10:11]                              // 00000003C1A0: 87FE0A7E
	v_lshl_add_u64 v[6:7], v[0:1], 1, v[34:35]                 // 00000003C1A4: D2080006 04890300
	v_lshl_add_u64 v[0:1], v[0:1], 0, 64                       // 00000003C1AC: D2080000 03010100
	v_cmp_ge_u64_e64 s[4:5], v[0:1], v[36:37]                  // 00000003C1B4: D0EE0004 00024900
	s_or_b64 s[8:9], s[4:5], s[8:9]                            // 00000003C1BC: 87880804
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[86:87]                 // 00000003C1C0: D2080002 01590102
	flat_store_short_d16_hi v[6:7], v5                         // 00000003C1C8: DC6C0000 00000506
	s_andn2_b64 exec, exec, s[8:9]                             // 00000003C1D0: 89FE087E
	s_cbranch_execz 29                                         // 00000003C1D4: BF88001D <EpCombineIntraNodeKernel_bf16_nop2p+0x324c>
	v_mov_b32_e32 v4, 0                                        // 00000003C1D8: 7E080280
	s_and_saveexec_b64 s[4:5], vcc                             // 00000003C1DC: BE84206A
	s_cbranch_execz 5                                          // 00000003C1E0: BF880005 <EpCombineIntraNodeKernel_bf16_nop2p+0x31f8>
	flat_load_ushort v4, v[2:3]                                // 00000003C1E4: DC480000 04000002
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003C1EC: BF8C0070
	v_lshlrev_b32_e32 v4, 16, v4                               // 00000003C1F0: 24080890
	v_add_f32_e32 v4, 0, v4                                    // 00000003C1F4: 02080880
	s_or_b64 exec, exec, s[4:5]                                // 00000003C1F8: 87FE047E
	v_and_b32_e32 v5, 0x7f800000, v4                           // 00000003C1FC: 260A08FF 7F800000
	v_cmp_ne_u32_e64 s[4:5], s58, v5                           // 00000003C204: D0CD0004 00020A3A
	s_and_saveexec_b64 s[10:11], s[4:5]                        // 00000003C20C: BE8A2004
	s_xor_b64 s[4:5], exec, s[10:11]                           // 00000003C210: 88840A7E
	v_bfe_u32 v5, v4, 16, 1                                    // 00000003C214: D1C80005 02052104
	v_add3_u32 v5, v4, v5, s59                                 // 00000003C21C: D1FF0005 00EE0B04
	s_andn2_saveexec_b64 s[10:11], s[4:5]                      // 00000003C224: BE8A2304
	s_cbranch_execz 65501                                      // 00000003C228: BF88FFDD <EpCombineIntraNodeKernel_bf16_nop2p+0x31a0>
	v_or_b32_e32 v5, 0x10000, v4                               // 00000003C22C: 280A08FF 00010000
	v_cmp_eq_u32_sdwa s[4:5], v4, v21 src0_sel:WORD_0 src1_sel:DWORD// 00000003C234: 7D942AF9 06048404
	s_nop 1                                                    // 00000003C23C: BF800001
	v_cndmask_b32_e64 v5, v5, v4, s[4:5]                       // 00000003C240: D1000005 00120905
	s_branch 65493                                             // 00000003C248: BF82FFD5 <EpCombineIntraNodeKernel_bf16_nop2p+0x31a0>
	s_or_b64 exec, exec, s[6:7]                                // 00000003C24C: 87FE067E
	v_cmp_eq_u32_e32 vcc, s51, v33                             // 00000003C250: 7D944233
	s_and_b64 s[4:5], s[62:63], vcc                            // 00000003C254: 86846A3E
	s_and_saveexec_b64 s[24:25], s[4:5]                        // 00000003C258: BE982004
	s_cbranch_execz 63114                                      // 00000003C25C: BF88F68A <EpCombineIntraNodeKernel_bf16_nop2p+0xc88>
	v_readlane_b32 s4, v63, 4                                  // 00000003C260: D2890004 0001093F
	v_readlane_b32 s5, v63, 5                                  // 00000003C268: D2890005 00010B3F
	v_mul_lo_u32 v32, v32, s46                                 // 00000003C270: D2850020 00005D20
	v_ashrrev_i32_e32 v33, 31, v32                             // 00000003C278: 2242409F
	s_mov_b64 s[6:7], -1                                       // 00000003C27C: BE8601C1
	s_mov_b64 s[26:27], 0                                      // 00000003C280: BE9A0180
	s_cmp_lt_i32 s46, 6                                        // 00000003C284: BF04862E
	global_load_dwordx2 v[34:35], v21, s[4:5]                  // 00000003C288: DC548000 22040015
	s_mov_b64 s[4:5], 0                                        // 00000003C290: BE840180
	s_waitcnt vmcnt(0)                                         // 00000003C294: BF8C0F70
	v_lshl_add_u64 v[36:37], v[32:33], 2, v[34:35]             // 00000003C298: D2080024 04890520
	s_cbranch_scc1 444                                         // 00000003C2A0: BF8501BC <EpCombineIntraNodeKernel_bf16_nop2p+0x3994>
	s_cmp_gt_i32 s46, 7                                        // 00000003C2A4: BF02872E
	s_cbranch_scc0 322                                         // 00000003C2A8: BF840142 <EpCombineIntraNodeKernel_bf16_nop2p+0x37b4>
	s_cmp_gt_i32 s46, 9                                        // 00000003C2AC: BF02892E
	s_cbranch_scc0 173                                         // 00000003C2B0: BF8400AD <EpCombineIntraNodeKernel_bf16_nop2p+0x3568>
	s_cmp_eq_u32 s46, 10                                       // 00000003C2B4: BF068A2E
	s_mov_b64 s[4:5], -1                                       // 00000003C2B8: BE8401C1
	s_cbranch_scc0 169                                         // 00000003C2BC: BF8400A9 <EpCombineIntraNodeKernel_bf16_nop2p+0x3564>
	s_mov_b64 s[28:29], exec                                   // 00000003C2C0: BE9C017E
	v_readlane_b32 s4, v63, 9                                  // 00000003C2C4: D2890004 0001133F
	v_readlane_b32 s5, v63, 10                                 // 00000003C2CC: D2890005 0001153F
	s_and_b64 s[4:5], s[28:29], s[4:5]                         // 00000003C2D4: 8684041C
	s_mov_b64 exec, s[4:5]                                     // 00000003C2D8: BEFE0104
	s_cbranch_execz 159                                        // 00000003C2DC: BF88009F <EpCombineIntraNodeKernel_bf16_nop2p+0x355c>
	ds_read2_b64 v[0:3], v29 offset1:1                         // 00000003C2E0: D8EE0100 0000001D
	ds_read2_b64 v[4:7], v29 offset0:2 offset1:3               // 00000003C2E8: D8EE0302 0400001D
	ds_read2_b64 v[8:11], v29 offset0:4 offset1:5              // 00000003C2F0: D8EE0504 0800001D
	ds_read2_b64 v[12:15], v29 offset0:6 offset1:7             // 00000003C2F8: D8EE0706 0C00001D
	ds_read2_b64 v[16:19], v29 offset0:8 offset1:9             // 00000003C300: D8EE0908 1000001D
	s_mov_b64 s[30:31], 0                                      // 00000003C308: BE9E0180
	s_waitcnt lgkmcnt(0)                                       // 00000003C30C: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[0:1]                            // 00000003C310: 7DDA0080
	v_cmp_ne_u64_e64 s[4:5], 0, v[2:3]                         // 00000003C314: D0ED0004 00020480
	v_cmp_ne_u64_e64 s[6:7], 0, v[4:5]                         // 00000003C31C: D0ED0006 00020880
	v_cmp_ne_u64_e64 s[8:9], 0, v[6:7]                         // 00000003C324: D0ED0008 00020C80
	v_cmp_ne_u64_e64 s[10:11], 0, v[8:9]                       // 00000003C32C: D0ED000A 00021080
	v_cmp_ne_u64_e64 s[12:13], 0, v[10:11]                     // 00000003C334: D0ED000C 00021480
	v_cmp_ne_u64_e64 s[14:15], 0, v[12:13]                     // 00000003C33C: D0ED000E 00021880
	v_cmp_ne_u64_e64 s[16:17], 0, v[14:15]                     // 00000003C344: D0ED0010 00021C80
	v_cmp_ne_u64_e64 s[18:19], 0, v[16:17]                     // 00000003C34C: D0ED0012 00022080
	v_cmp_ne_u64_e64 s[20:21], 0, v[18:19]                     // 00000003C354: D0ED0014 00022480
	v_mov_b64_e32 v[38:39], v[36:37]                           // 00000003C35C: 7E4C7124
	v_mov_b64_e32 v[40:41], v[42:43]                           // 00000003C360: 7E50712A
	s_branch 34                                                // 00000003C364: BF820022 <EpCombineIntraNodeKernel_bf16_nop2p+0x33f0>
	s_or_b64 exec, exec, s[22:23]                              // 00000003C368: 87FE167E
	v_lshl_add_u64 v[40:41], v[40:41], 0, 64                   // 00000003C36C: D2080028 03010128
	v_cmp_le_u64_e64 s[22:23], s[46:47], v[40:41]              // 00000003C374: D0EB0016 0002502E
	v_lshl_add_u64 v[44:45], v[38:39], 0, v[30:31]             // 00000003C37C: D208002C 04790126
	v_lshl_add_u64 v[38:39], v[38:39], 0, s[88:89]             // 00000003C384: D2080026 01610126
	v_lshl_add_u64 v[18:19], v[18:19], 0, s[88:89]             // 00000003C38C: D2080012 01610112
	v_lshl_add_u64 v[16:17], v[16:17], 0, s[88:89]             // 00000003C394: D2080010 01610110
	v_lshl_add_u64 v[14:15], v[14:15], 0, s[88:89]             // 00000003C39C: D208000E 0161010E
	v_lshl_add_u64 v[12:13], v[12:13], 0, s[88:89]             // 00000003C3A4: D208000C 0161010C
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[88:89]             // 00000003C3AC: D208000A 0161010A
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[88:89]                 // 00000003C3B4: D2080008 01610108
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[88:89]                 // 00000003C3BC: D2080006 01610106
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[88:89]                 // 00000003C3C4: D2080004 01610104
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[88:89]                 // 00000003C3CC: D2080002 01610102
	s_or_b64 s[30:31], s[22:23], s[30:31]                      // 00000003C3D4: 879E1E16
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[88:89]                 // 00000003C3D8: D2080000 01610100
	flat_store_dword v[44:45], v20                             // 00000003C3E0: DC700000 0000142C
	s_andn2_b64 exec, exec, s[30:31]                           // 00000003C3E8: 89FE1E7E
	s_cbranch_execz 91                                         // 00000003C3EC: BF88005B <EpCombineIntraNodeKernel_bf16_nop2p+0x355c>
	v_mov_b32_e32 v20, 0                                       // 00000003C3F0: 7E280280
	s_and_saveexec_b64 s[22:23], vcc                           // 00000003C3F4: BE96206A
	s_cbranch_execz 6                                          // 00000003C3F8: BF880006 <EpCombineIntraNodeKernel_bf16_nop2p+0x3414>
	v_lshl_add_u64 v[44:45], v[0:1], 0, v[30:31]               // 00000003C3FC: D208002C 04790100
	flat_load_dword v20, v[44:45]                              // 00000003C404: DC500000 1400002C
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003C40C: BF8C0070
	v_add_f32_e32 v20, 0, v20                                  // 00000003C410: 02282880
	s_or_b64 exec, exec, s[22:23]                              // 00000003C414: 87FE167E
	s_and_saveexec_b64 s[22:23], s[4:5]                        // 00000003C418: BE962004
	s_cbranch_execz 6                                          // 00000003C41C: BF880006 <EpCombineIntraNodeKernel_bf16_nop2p+0x3438>
	v_lshl_add_u64 v[44:45], v[2:3], 0, v[30:31]               // 00000003C420: D208002C 04790102
	flat_load_dword v44, v[44:45]                              // 00000003C428: DC500000 2C00002C
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003C430: BF8C0070
	v_add_f32_e32 v20, v20, v44                                // 00000003C434: 02285914
	s_or_b64 exec, exec, s[22:23]                              // 00000003C438: 87FE167E
	s_and_saveexec_b64 s[22:23], s[6:7]                        // 00000003C43C: BE962006
	s_cbranch_execz 6                                          // 00000003C440: BF880006 <EpCombineIntraNodeKernel_bf16_nop2p+0x345c>
	v_lshl_add_u64 v[44:45], v[4:5], 0, v[30:31]               // 00000003C444: D208002C 04790104
	flat_load_dword v44, v[44:45]                              // 00000003C44C: DC500000 2C00002C
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003C454: BF8C0070
	v_add_f32_e32 v20, v20, v44                                // 00000003C458: 02285914
	s_or_b64 exec, exec, s[22:23]                              // 00000003C45C: 87FE167E
	s_and_saveexec_b64 s[22:23], s[8:9]                        // 00000003C460: BE962008
	s_cbranch_execz 6                                          // 00000003C464: BF880006 <EpCombineIntraNodeKernel_bf16_nop2p+0x3480>
	v_lshl_add_u64 v[44:45], v[6:7], 0, v[30:31]               // 00000003C468: D208002C 04790106
	flat_load_dword v44, v[44:45]                              // 00000003C470: DC500000 2C00002C
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003C478: BF8C0070
	v_add_f32_e32 v20, v20, v44                                // 00000003C47C: 02285914
	s_or_b64 exec, exec, s[22:23]                              // 00000003C480: 87FE167E
	s_and_saveexec_b64 s[22:23], s[10:11]                      // 00000003C484: BE96200A
	s_cbranch_execz 6                                          // 00000003C488: BF880006 <EpCombineIntraNodeKernel_bf16_nop2p+0x34a4>
	v_lshl_add_u64 v[44:45], v[8:9], 0, v[30:31]               // 00000003C48C: D208002C 04790108
	flat_load_dword v44, v[44:45]                              // 00000003C494: DC500000 2C00002C
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003C49C: BF8C0070
	v_add_f32_e32 v20, v20, v44                                // 00000003C4A0: 02285914
	s_or_b64 exec, exec, s[22:23]                              // 00000003C4A4: 87FE167E
	s_and_saveexec_b64 s[22:23], s[12:13]                      // 00000003C4A8: BE96200C
	s_cbranch_execz 6                                          // 00000003C4AC: BF880006 <EpCombineIntraNodeKernel_bf16_nop2p+0x34c8>
	v_lshl_add_u64 v[44:45], v[10:11], 0, v[30:31]             // 00000003C4B0: D208002C 0479010A
	flat_load_dword v44, v[44:45]                              // 00000003C4B8: DC500000 2C00002C
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003C4C0: BF8C0070
	v_add_f32_e32 v20, v20, v44                                // 00000003C4C4: 02285914
	s_or_b64 exec, exec, s[22:23]                              // 00000003C4C8: 87FE167E
	s_and_saveexec_b64 s[22:23], s[14:15]                      // 00000003C4CC: BE96200E
	s_cbranch_execz 6                                          // 00000003C4D0: BF880006 <EpCombineIntraNodeKernel_bf16_nop2p+0x34ec>
	v_lshl_add_u64 v[44:45], v[12:13], 0, v[30:31]             // 00000003C4D4: D208002C 0479010C
	flat_load_dword v44, v[44:45]                              // 00000003C4DC: DC500000 2C00002C
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003C4E4: BF8C0070
	v_add_f32_e32 v20, v20, v44                                // 00000003C4E8: 02285914
	s_or_b64 exec, exec, s[22:23]                              // 00000003C4EC: 87FE167E
	s_and_saveexec_b64 s[22:23], s[16:17]                      // 00000003C4F0: BE962010
	s_cbranch_execz 6                                          // 00000003C4F4: BF880006 <EpCombineIntraNodeKernel_bf16_nop2p+0x3510>
	v_lshl_add_u64 v[44:45], v[14:15], 0, v[30:31]             // 00000003C4F8: D208002C 0479010E
	flat_load_dword v44, v[44:45]                              // 00000003C500: DC500000 2C00002C
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003C508: BF8C0070
	v_add_f32_e32 v20, v20, v44                                // 00000003C50C: 02285914
	s_or_b64 exec, exec, s[22:23]                              // 00000003C510: 87FE167E
	s_and_saveexec_b64 s[22:23], s[18:19]                      // 00000003C514: BE962012
	s_cbranch_execz 6                                          // 00000003C518: BF880006 <EpCombineIntraNodeKernel_bf16_nop2p+0x3534>
	v_lshl_add_u64 v[44:45], v[16:17], 0, v[30:31]             // 00000003C51C: D208002C 04790110
	flat_load_dword v44, v[44:45]                              // 00000003C524: DC500000 2C00002C
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003C52C: BF8C0070
	v_add_f32_e32 v20, v20, v44                                // 00000003C530: 02285914
	s_or_b64 exec, exec, s[22:23]                              // 00000003C534: 87FE167E
	s_and_saveexec_b64 s[22:23], s[20:21]                      // 00000003C538: BE962014
	s_cbranch_execz 65418                                      // 00000003C53C: BF88FF8A <EpCombineIntraNodeKernel_bf16_nop2p+0x3368>
	v_lshl_add_u64 v[44:45], v[18:19], 0, v[30:31]             // 00000003C540: D208002C 04790112
	flat_load_dword v44, v[44:45]                              // 00000003C548: DC500000 2C00002C
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003C550: BF8C0070
	v_add_f32_e32 v20, v20, v44                                // 00000003C554: 02285914
	s_branch 65411                                             // 00000003C558: BF82FF83 <EpCombineIntraNodeKernel_bf16_nop2p+0x3368>
	s_or_b64 exec, exec, s[28:29]                              // 00000003C55C: 87FE1C7E
	s_mov_b64 s[4:5], 0                                        // 00000003C560: BE840180
	s_mov_b64 s[6:7], 0                                        // 00000003C564: BE860180
	s_and_b64 vcc, exec, s[6:7]                                // 00000003C568: 86EA067E
	s_cbranch_vccz 144                                         // 00000003C56C: BF860090 <EpCombineIntraNodeKernel_bf16_nop2p+0x37b0>
	s_cmp_eq_u32 s46, 8                                        // 00000003C570: BF06882E
	s_mov_b64 s[4:5], -1                                       // 00000003C574: BE8401C1
	s_cbranch_scc0 141                                         // 00000003C578: BF84008D <EpCombineIntraNodeKernel_bf16_nop2p+0x37b0>
	s_mov_b64 s[20:21], exec                                   // 00000003C57C: BE94017E
	v_readlane_b32 s4, v63, 9                                  // 00000003C580: D2890004 0001133F
	v_readlane_b32 s5, v63, 10                                 // 00000003C588: D2890005 0001153F
	s_and_b64 s[4:5], s[20:21], s[4:5]                         // 00000003C590: 86840414
	s_mov_b64 exec, s[4:5]                                     // 00000003C594: BEFE0104
	s_cbranch_execz 131                                        // 00000003C598: BF880083 <EpCombineIntraNodeKernel_bf16_nop2p+0x37a8>
	ds_read2_b64 v[0:3], v29 offset1:1                         // 00000003C59C: D8EE0100 0000001D
	ds_read2_b64 v[4:7], v29 offset0:2 offset1:3               // 00000003C5A4: D8EE0302 0400001D
	ds_read2_b64 v[8:11], v29 offset0:4 offset1:5              // 00000003C5AC: D8EE0504 0800001D
	ds_read2_b64 v[12:15], v29 offset0:6 offset1:7             // 00000003C5B4: D8EE0706 0C00001D
	s_mov_b64 s[22:23], 0                                      // 00000003C5BC: BE960180
	v_mov_b64_e32 v[16:17], v[36:37]                           // 00000003C5C0: 7E207124
	s_waitcnt lgkmcnt(0)                                       // 00000003C5C4: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[0:1]                            // 00000003C5C8: 7DDA0080
	v_cmp_ne_u64_e64 s[4:5], 0, v[2:3]                         // 00000003C5CC: D0ED0004 00020480
	v_cmp_ne_u64_e64 s[6:7], 0, v[4:5]                         // 00000003C5D4: D0ED0006 00020880
	v_cmp_ne_u64_e64 s[8:9], 0, v[6:7]                         // 00000003C5DC: D0ED0008 00020C80
	v_cmp_ne_u64_e64 s[10:11], 0, v[8:9]                       // 00000003C5E4: D0ED000A 00021080
	v_cmp_ne_u64_e64 s[12:13], 0, v[10:11]                     // 00000003C5EC: D0ED000C 00021480
	v_cmp_ne_u64_e64 s[14:15], 0, v[12:13]                     // 00000003C5F4: D0ED000E 00021880
	v_cmp_ne_u64_e64 s[16:17], 0, v[14:15]                     // 00000003C5FC: D0ED0010 00021C80
	v_mov_b64_e32 v[18:19], v[42:43]                           // 00000003C604: 7E24712A
	s_branch 30                                                // 00000003C608: BF82001E <EpCombineIntraNodeKernel_bf16_nop2p+0x3684>
	s_or_b64 exec, exec, s[18:19]                              // 00000003C60C: 87FE127E
	v_lshl_add_u64 v[18:19], v[18:19], 0, 64                   // 00000003C610: D2080012 03010112
	v_cmp_le_u64_e64 s[18:19], s[46:47], v[18:19]              // 00000003C618: D0EB0012 0002242E
	v_lshl_add_u64 v[38:39], v[16:17], 0, v[30:31]             // 00000003C620: D2080026 04790110
	v_lshl_add_u64 v[16:17], v[16:17], 0, s[88:89]             // 00000003C628: D2080010 01610110
	v_lshl_add_u64 v[14:15], v[14:15], 0, s[88:89]             // 00000003C630: D208000E 0161010E
	v_lshl_add_u64 v[12:13], v[12:13], 0, s[88:89]             // 00000003C638: D208000C 0161010C
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[88:89]             // 00000003C640: D208000A 0161010A
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[88:89]                 // 00000003C648: D2080008 01610108
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[88:89]                 // 00000003C650: D2080006 01610106
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[88:89]                 // 00000003C658: D2080004 01610104
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[88:89]                 // 00000003C660: D2080002 01610102
	s_or_b64 s[22:23], s[18:19], s[22:23]                      // 00000003C668: 87961612
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[88:89]                 // 00000003C66C: D2080000 01610100
	flat_store_dword v[38:39], v20                             // 00000003C674: DC700000 00001426
	s_andn2_b64 exec, exec, s[22:23]                           // 00000003C67C: 89FE167E
	s_cbranch_execz 73                                         // 00000003C680: BF880049 <EpCombineIntraNodeKernel_bf16_nop2p+0x37a8>
	v_mov_b32_e32 v20, 0                                       // 00000003C684: 7E280280
	s_and_saveexec_b64 s[18:19], vcc                           // 00000003C688: BE92206A
	s_cbranch_execz 6                                          // 00000003C68C: BF880006 <EpCombineIntraNodeKernel_bf16_nop2p+0x36a8>
	v_lshl_add_u64 v[38:39], v[0:1], 0, v[30:31]               // 00000003C690: D2080026 04790100
	flat_load_dword v20, v[38:39]                              // 00000003C698: DC500000 14000026
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003C6A0: BF8C0070
	v_add_f32_e32 v20, 0, v20                                  // 00000003C6A4: 02282880
	s_or_b64 exec, exec, s[18:19]                              // 00000003C6A8: 87FE127E
	s_and_saveexec_b64 s[18:19], s[4:5]                        // 00000003C6AC: BE922004
	s_cbranch_execz 6                                          // 00000003C6B0: BF880006 <EpCombineIntraNodeKernel_bf16_nop2p+0x36cc>
	v_lshl_add_u64 v[38:39], v[2:3], 0, v[30:31]               // 00000003C6B4: D2080026 04790102
	flat_load_dword v38, v[38:39]                              // 00000003C6BC: DC500000 26000026
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003C6C4: BF8C0070
	v_add_f32_e32 v20, v20, v38                                // 00000003C6C8: 02284D14
	s_or_b64 exec, exec, s[18:19]                              // 00000003C6CC: 87FE127E
	s_and_saveexec_b64 s[18:19], s[6:7]                        // 00000003C6D0: BE922006
	s_cbranch_execz 6                                          // 00000003C6D4: BF880006 <EpCombineIntraNodeKernel_bf16_nop2p+0x36f0>
	v_lshl_add_u64 v[38:39], v[4:5], 0, v[30:31]               // 00000003C6D8: D2080026 04790104
	flat_load_dword v38, v[38:39]                              // 00000003C6E0: DC500000 26000026
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003C6E8: BF8C0070
	v_add_f32_e32 v20, v20, v38                                // 00000003C6EC: 02284D14
	s_or_b64 exec, exec, s[18:19]                              // 00000003C6F0: 87FE127E
	s_and_saveexec_b64 s[18:19], s[8:9]                        // 00000003C6F4: BE922008
	s_cbranch_execz 6                                          // 00000003C6F8: BF880006 <EpCombineIntraNodeKernel_bf16_nop2p+0x3714>
	v_lshl_add_u64 v[38:39], v[6:7], 0, v[30:31]               // 00000003C6FC: D2080026 04790106
	flat_load_dword v38, v[38:39]                              // 00000003C704: DC500000 26000026
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003C70C: BF8C0070
	v_add_f32_e32 v20, v20, v38                                // 00000003C710: 02284D14
	s_or_b64 exec, exec, s[18:19]                              // 00000003C714: 87FE127E
	s_and_saveexec_b64 s[18:19], s[10:11]                      // 00000003C718: BE92200A
	s_cbranch_execz 6                                          // 00000003C71C: BF880006 <EpCombineIntraNodeKernel_bf16_nop2p+0x3738>
	v_lshl_add_u64 v[38:39], v[8:9], 0, v[30:31]               // 00000003C720: D2080026 04790108
	flat_load_dword v38, v[38:39]                              // 00000003C728: DC500000 26000026
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003C730: BF8C0070
	v_add_f32_e32 v20, v20, v38                                // 00000003C734: 02284D14
	s_or_b64 exec, exec, s[18:19]                              // 00000003C738: 87FE127E
	s_and_saveexec_b64 s[18:19], s[12:13]                      // 00000003C73C: BE92200C
	s_cbranch_execz 6                                          // 00000003C740: BF880006 <EpCombineIntraNodeKernel_bf16_nop2p+0x375c>
	v_lshl_add_u64 v[38:39], v[10:11], 0, v[30:31]             // 00000003C744: D2080026 0479010A
	flat_load_dword v38, v[38:39]                              // 00000003C74C: DC500000 26000026
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003C754: BF8C0070
	v_add_f32_e32 v20, v20, v38                                // 00000003C758: 02284D14
	s_or_b64 exec, exec, s[18:19]                              // 00000003C75C: 87FE127E
	s_and_saveexec_b64 s[18:19], s[14:15]                      // 00000003C760: BE92200E
	s_cbranch_execz 6                                          // 00000003C764: BF880006 <EpCombineIntraNodeKernel_bf16_nop2p+0x3780>
	v_lshl_add_u64 v[38:39], v[12:13], 0, v[30:31]             // 00000003C768: D2080026 0479010C
	flat_load_dword v38, v[38:39]                              // 00000003C770: DC500000 26000026
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003C778: BF8C0070
	v_add_f32_e32 v20, v20, v38                                // 00000003C77C: 02284D14
	s_or_b64 exec, exec, s[18:19]                              // 00000003C780: 87FE127E
	s_and_saveexec_b64 s[18:19], s[16:17]                      // 00000003C784: BE922010
	s_cbranch_execz 65440                                      // 00000003C788: BF88FFA0 <EpCombineIntraNodeKernel_bf16_nop2p+0x360c>
	v_lshl_add_u64 v[38:39], v[14:15], 0, v[30:31]             // 00000003C78C: D2080026 0479010E
	flat_load_dword v38, v[38:39]                              // 00000003C794: DC500000 26000026
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003C79C: BF8C0070
	v_add_f32_e32 v20, v20, v38                                // 00000003C7A0: 02284D14
	s_branch 65433                                             // 00000003C7A4: BF82FF99 <EpCombineIntraNodeKernel_bf16_nop2p+0x360c>
	s_or_b64 exec, exec, s[20:21]                              // 00000003C7A8: 87FE147E
	s_mov_b64 s[4:5], 0                                        // 00000003C7AC: BE840180
	s_mov_b64 s[6:7], 0                                        // 00000003C7B0: BE860180
	s_and_b64 vcc, exec, s[6:7]                                // 00000003C7B4: 86EA067E
	s_cbranch_vccz 117                                         // 00000003C7B8: BF860075 <EpCombineIntraNodeKernel_bf16_nop2p+0x3990>
	s_cmp_eq_u32 s46, 6                                        // 00000003C7BC: BF06862E
	s_mov_b64 s[4:5], -1                                       // 00000003C7C0: BE8401C1
	s_cbranch_scc0 114                                         // 00000003C7C4: BF840072 <EpCombineIntraNodeKernel_bf16_nop2p+0x3990>
	s_mov_b64 s[16:17], exec                                   // 00000003C7C8: BE90017E
	v_readlane_b32 s4, v63, 9                                  // 00000003C7CC: D2890004 0001133F
	v_readlane_b32 s5, v63, 10                                 // 00000003C7D4: D2890005 0001153F
	s_and_b64 s[4:5], s[16:17], s[4:5]                         // 00000003C7DC: 86840410
	s_mov_b64 exec, s[4:5]                                     // 00000003C7E0: BEFE0104
	s_cbranch_execz 104                                        // 00000003C7E4: BF880068 <EpCombineIntraNodeKernel_bf16_nop2p+0x3988>
	ds_read2_b64 v[12:15], v29 offset1:1                       // 00000003C7E8: D8EE0100 0C00001D
	ds_read2_b64 v[8:11], v29 offset0:2 offset1:3              // 00000003C7F0: D8EE0302 0800001D
	ds_read2_b64 v[4:7], v29 offset0:4 offset1:5               // 00000003C7F8: D8EE0504 0400001D
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[30:31]               // 00000003C800: D2080000 04790122
	s_mov_b64 s[18:19], 0                                      // 00000003C808: BE920180
	s_waitcnt lgkmcnt(0)                                       // 00000003C80C: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[12:13]                          // 00000003C810: 7DDA1880
	v_cmp_ne_u64_e64 s[4:5], 0, v[14:15]                       // 00000003C814: D0ED0004 00021C80
	v_cmp_ne_u64_e64 s[6:7], 0, v[8:9]                         // 00000003C81C: D0ED0006 00021080
	v_cmp_ne_u64_e64 s[8:9], 0, v[10:11]                       // 00000003C824: D0ED0008 00021480
	v_cmp_ne_u64_e64 s[10:11], 0, v[4:5]                       // 00000003C82C: D0ED000A 00020880
	v_cmp_ne_u64_e64 s[12:13], 0, v[6:7]                       // 00000003C834: D0ED000C 00020C80
	v_lshl_add_u64 v[0:1], v[32:33], 2, v[0:1]                 // 00000003C83C: D2080000 04010520
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[30:31]                 // 00000003C844: D2080002 04790106
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[30:31]                 // 00000003C84C: D2080004 04790104
	v_lshl_add_u64 v[6:7], v[10:11], 0, v[30:31]               // 00000003C854: D2080006 0479010A
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[30:31]                 // 00000003C85C: D2080008 04790108
	v_lshl_add_u64 v[10:11], v[14:15], 0, v[30:31]             // 00000003C864: D208000A 0479010E
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[30:31]             // 00000003C86C: D208000C 0479010C
	v_mov_b64_e32 v[14:15], v[42:43]                           // 00000003C874: 7E1C712A
	s_branch 24                                                // 00000003C878: BF820018 <EpCombineIntraNodeKernel_bf16_nop2p+0x38dc>
	s_or_b64 exec, exec, s[14:15]                              // 00000003C87C: 87FE0E7E
	v_lshl_add_u64 v[14:15], v[14:15], 0, 64                   // 00000003C880: D208000E 0301010E
	v_cmp_le_u64_e64 s[14:15], s[46:47], v[14:15]              // 00000003C888: D0EB000E 00021C2E
	flat_store_dword v[0:1], v16                               // 00000003C890: DC700000 00001000
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[88:89]                 // 00000003C898: D2080000 01610100
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[88:89]                 // 00000003C8A0: D2080002 01610102
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[88:89]                 // 00000003C8A8: D2080004 01610104
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[88:89]                 // 00000003C8B0: D2080006 01610106
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[88:89]                 // 00000003C8B8: D2080008 01610108
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[88:89]             // 00000003C8C0: D208000A 0161010A
	s_or_b64 s[18:19], s[14:15], s[18:19]                      // 00000003C8C8: 8792120E
	v_lshl_add_u64 v[12:13], v[12:13], 0, s[88:89]             // 00000003C8CC: D208000C 0161010C
	s_andn2_b64 exec, exec, s[18:19]                           // 00000003C8D4: 89FE127E
	s_cbranch_execz 43                                         // 00000003C8D8: BF88002B <EpCombineIntraNodeKernel_bf16_nop2p+0x3988>
	v_mov_b32_e32 v16, 0                                       // 00000003C8DC: 7E200280
	s_and_saveexec_b64 s[14:15], vcc                           // 00000003C8E0: BE8E206A
	s_cbranch_execz 4                                          // 00000003C8E4: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x38f8>
	flat_load_dword v16, v[12:13]                              // 00000003C8E8: DC500000 1000000C
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003C8F0: BF8C0070
	v_add_f32_e32 v16, 0, v16                                  // 00000003C8F4: 02202080
	s_or_b64 exec, exec, s[14:15]                              // 00000003C8F8: 87FE0E7E
	s_and_saveexec_b64 s[14:15], s[4:5]                        // 00000003C8FC: BE8E2004
	s_cbranch_execz 4                                          // 00000003C900: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x3914>
	flat_load_dword v17, v[10:11]                              // 00000003C904: DC500000 1100000A
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003C90C: BF8C0070
	v_add_f32_e32 v16, v16, v17                                // 00000003C910: 02202310
	s_or_b64 exec, exec, s[14:15]                              // 00000003C914: 87FE0E7E
	s_and_saveexec_b64 s[14:15], s[6:7]                        // 00000003C918: BE8E2006
	s_cbranch_execz 4                                          // 00000003C91C: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x3930>
	flat_load_dword v17, v[8:9]                                // 00000003C920: DC500000 11000008
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003C928: BF8C0070
	v_add_f32_e32 v16, v16, v17                                // 00000003C92C: 02202310
	s_or_b64 exec, exec, s[14:15]                              // 00000003C930: 87FE0E7E
	s_and_saveexec_b64 s[14:15], s[8:9]                        // 00000003C934: BE8E2008
	s_cbranch_execz 4                                          // 00000003C938: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x394c>
	flat_load_dword v17, v[6:7]                                // 00000003C93C: DC500000 11000006
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003C944: BF8C0070
	v_add_f32_e32 v16, v16, v17                                // 00000003C948: 02202310
	s_or_b64 exec, exec, s[14:15]                              // 00000003C94C: 87FE0E7E
	s_and_saveexec_b64 s[14:15], s[10:11]                      // 00000003C950: BE8E200A
	s_cbranch_execz 4                                          // 00000003C954: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x3968>
	flat_load_dword v17, v[4:5]                                // 00000003C958: DC500000 11000004
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003C960: BF8C0070
	v_add_f32_e32 v16, v16, v17                                // 00000003C964: 02202310
	s_or_b64 exec, exec, s[14:15]                              // 00000003C968: 87FE0E7E
	s_and_saveexec_b64 s[14:15], s[12:13]                      // 00000003C96C: BE8E200C
	s_cbranch_execz 65474                                      // 00000003C970: BF88FFC2 <EpCombineIntraNodeKernel_bf16_nop2p+0x387c>
	flat_load_dword v17, v[2:3]                                // 00000003C974: DC500000 11000002
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003C97C: BF8C0070
	v_add_f32_e32 v16, v16, v17                                // 00000003C980: 02202310
	s_branch 65469                                             // 00000003C984: BF82FFBD <EpCombineIntraNodeKernel_bf16_nop2p+0x387c>
	s_or_b64 exec, exec, s[16:17]                              // 00000003C988: 87FE107E
	s_mov_b64 s[4:5], 0                                        // 00000003C98C: BE840180
	s_mov_b64 s[6:7], 0                                        // 00000003C990: BE860180
	s_and_b64 vcc, exec, s[6:7]                                // 00000003C994: 86EA067E
	s_cbranch_vccz 165                                         // 00000003C998: BF8600A5 <EpCombineIntraNodeKernel_bf16_nop2p+0x3c30>
	s_cmp_gt_i32 s46, 1                                        // 00000003C99C: BF02812E
	s_mov_b64 s[6:7], -1                                       // 00000003C9A0: BE8601C1
	s_cbranch_scc0 156                                         // 00000003C9A4: BF84009C <EpCombineIntraNodeKernel_bf16_nop2p+0x3c18>
	s_cmp_gt_i32 s46, 3                                        // 00000003C9A8: BF02832E
	s_cbranch_scc0 90                                          // 00000003C9AC: BF84005A <EpCombineIntraNodeKernel_bf16_nop2p+0x3b18>
	s_cmp_eq_u32 s46, 4                                        // 00000003C9B0: BF06842E
	s_mov_b64 s[4:5], -1                                       // 00000003C9B4: BE8401C1
	s_cbranch_scc0 86                                          // 00000003C9B8: BF840056 <EpCombineIntraNodeKernel_bf16_nop2p+0x3b14>
	s_mov_b64 s[12:13], exec                                   // 00000003C9BC: BE8C017E
	v_readlane_b32 s4, v63, 9                                  // 00000003C9C0: D2890004 0001133F
	v_readlane_b32 s5, v63, 10                                 // 00000003C9C8: D2890005 0001153F
	s_and_b64 s[4:5], s[12:13], s[4:5]                         // 00000003C9D0: 8684040C
	s_mov_b64 exec, s[4:5]                                     // 00000003C9D4: BEFE0104
	s_cbranch_execz 76                                         // 00000003C9D8: BF88004C <EpCombineIntraNodeKernel_bf16_nop2p+0x3b0c>
	ds_read2_b64 v[8:11], v29 offset1:1                        // 00000003C9DC: D8EE0100 0800001D
	ds_read2_b64 v[4:7], v29 offset0:2 offset1:3               // 00000003C9E4: D8EE0302 0400001D
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[30:31]               // 00000003C9EC: D2080000 04790122
	s_mov_b64 s[14:15], 0                                      // 00000003C9F4: BE8E0180
	v_lshl_add_u64 v[0:1], v[32:33], 2, v[0:1]                 // 00000003C9F8: D2080000 04010520
	s_waitcnt lgkmcnt(0)                                       // 00000003CA00: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[8:9]                            // 00000003CA04: 7DDA1080
	v_cmp_ne_u64_e64 s[4:5], 0, v[10:11]                       // 00000003CA08: D0ED0004 00021480
	v_cmp_ne_u64_e64 s[6:7], 0, v[4:5]                         // 00000003CA10: D0ED0006 00020880
	v_cmp_ne_u64_e64 s[8:9], 0, v[6:7]                         // 00000003CA18: D0ED0008 00020C80
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[30:31]                 // 00000003CA20: D2080002 04790106
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[30:31]                 // 00000003CA28: D2080004 04790104
	v_lshl_add_u64 v[6:7], v[10:11], 0, v[30:31]               // 00000003CA30: D2080006 0479010A
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[30:31]                 // 00000003CA38: D2080008 04790108
	v_mov_b64_e32 v[10:11], v[42:43]                           // 00000003CA40: 7E14712A
	s_branch 20                                                // 00000003CA44: BF820014 <EpCombineIntraNodeKernel_bf16_nop2p+0x3a98>
	s_or_b64 exec, exec, s[10:11]                              // 00000003CA48: 87FE0A7E
	v_lshl_add_u64 v[10:11], v[10:11], 0, 64                   // 00000003CA4C: D208000A 0301010A
	v_cmp_le_u64_e64 s[10:11], s[46:47], v[10:11]              // 00000003CA54: D0EB000A 0002142E
	flat_store_dword v[0:1], v12                               // 00000003CA5C: DC700000 00000C00
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[88:89]                 // 00000003CA64: D2080000 01610100
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[88:89]                 // 00000003CA6C: D2080002 01610102
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[88:89]                 // 00000003CA74: D2080004 01610104
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[88:89]                 // 00000003CA7C: D2080006 01610106
	s_or_b64 s[14:15], s[10:11], s[14:15]                      // 00000003CA84: 878E0E0A
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[88:89]                 // 00000003CA88: D2080008 01610108
	s_andn2_b64 exec, exec, s[14:15]                           // 00000003CA90: 89FE0E7E
	s_cbranch_execz 29                                         // 00000003CA94: BF88001D <EpCombineIntraNodeKernel_bf16_nop2p+0x3b0c>
	v_mov_b32_e32 v12, 0                                       // 00000003CA98: 7E180280
	s_and_saveexec_b64 s[10:11], vcc                           // 00000003CA9C: BE8A206A
	s_cbranch_execz 4                                          // 00000003CAA0: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x3ab4>
	flat_load_dword v12, v[8:9]                                // 00000003CAA4: DC500000 0C000008
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003CAAC: BF8C0070
	v_add_f32_e32 v12, 0, v12                                  // 00000003CAB0: 02181880
	s_or_b64 exec, exec, s[10:11]                              // 00000003CAB4: 87FE0A7E
	s_and_saveexec_b64 s[10:11], s[4:5]                        // 00000003CAB8: BE8A2004
	s_cbranch_execz 4                                          // 00000003CABC: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x3ad0>
	flat_load_dword v13, v[6:7]                                // 00000003CAC0: DC500000 0D000006
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003CAC8: BF8C0070
	v_add_f32_e32 v12, v12, v13                                // 00000003CACC: 02181B0C
	s_or_b64 exec, exec, s[10:11]                              // 00000003CAD0: 87FE0A7E
	s_and_saveexec_b64 s[10:11], s[6:7]                        // 00000003CAD4: BE8A2006
	s_cbranch_execz 4                                          // 00000003CAD8: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x3aec>
	flat_load_dword v13, v[4:5]                                // 00000003CADC: DC500000 0D000004
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003CAE4: BF8C0070
	v_add_f32_e32 v12, v12, v13                                // 00000003CAE8: 02181B0C
	s_or_b64 exec, exec, s[10:11]                              // 00000003CAEC: 87FE0A7E
	s_and_saveexec_b64 s[10:11], s[8:9]                        // 00000003CAF0: BE8A2008
	s_cbranch_execz 65492                                      // 00000003CAF4: BF88FFD4 <EpCombineIntraNodeKernel_bf16_nop2p+0x3a48>
	flat_load_dword v13, v[2:3]                                // 00000003CAF8: DC500000 0D000002
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003CB00: BF8C0070
	v_add_f32_e32 v12, v12, v13                                // 00000003CB04: 02181B0C
	s_branch 65487                                             // 00000003CB08: BF82FFCF <EpCombineIntraNodeKernel_bf16_nop2p+0x3a48>
	s_or_b64 exec, exec, s[12:13]                              // 00000003CB0C: 87FE0C7E
	s_mov_b64 s[4:5], 0                                        // 00000003CB10: BE840180
	s_mov_b64 s[6:7], 0                                        // 00000003CB14: BE860180
	s_and_b64 vcc, exec, s[6:7]                                // 00000003CB18: 86EA067E
	s_cbranch_vccz 61                                          // 00000003CB1C: BF86003D <EpCombineIntraNodeKernel_bf16_nop2p+0x3c14>
	s_cmp_eq_u32 s46, 2                                        // 00000003CB20: BF06822E
	s_mov_b64 s[4:5], -1                                       // 00000003CB24: BE8401C1
	s_cbranch_scc0 58                                          // 00000003CB28: BF84003A <EpCombineIntraNodeKernel_bf16_nop2p+0x3c14>
	s_mov_b64 s[8:9], exec                                     // 00000003CB2C: BE88017E
	v_readlane_b32 s4, v63, 9                                  // 00000003CB30: D2890004 0001133F
	v_readlane_b32 s5, v63, 10                                 // 00000003CB38: D2890005 0001153F
	s_and_b64 s[4:5], s[8:9], s[4:5]                           // 00000003CB40: 86840408
	s_mov_b64 exec, s[4:5]                                     // 00000003CB44: BEFE0104
	s_cbranch_execz 48                                         // 00000003CB48: BF880030 <EpCombineIntraNodeKernel_bf16_nop2p+0x3c0c>
	ds_read2_b64 v[4:7], v29 offset1:1                         // 00000003CB4C: D8EE0100 0400001D
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[30:31]               // 00000003CB54: D2080000 04790122
	s_mov_b64 s[10:11], 0                                      // 00000003CB5C: BE8A0180
	v_lshl_add_u64 v[0:1], v[32:33], 2, v[0:1]                 // 00000003CB60: D2080000 04010520
	s_waitcnt lgkmcnt(0)                                       // 00000003CB68: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[4:5]                            // 00000003CB6C: 7DDA0880
	v_cmp_ne_u64_e64 s[4:5], 0, v[6:7]                         // 00000003CB70: D0ED0004 00020C80
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[30:31]                 // 00000003CB78: D2080002 04790106
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[30:31]                 // 00000003CB80: D2080004 04790104
	v_mov_b64_e32 v[6:7], v[42:43]                             // 00000003CB88: 7E0C712A
	s_branch 16                                                // 00000003CB8C: BF820010 <EpCombineIntraNodeKernel_bf16_nop2p+0x3bd0>
	s_or_b64 exec, exec, s[6:7]                                // 00000003CB90: 87FE067E
	v_lshl_add_u64 v[6:7], v[6:7], 0, 64                       // 00000003CB94: D2080006 03010106
	v_cmp_le_u64_e64 s[6:7], s[46:47], v[6:7]                  // 00000003CB9C: D0EB0006 00020C2E
	flat_store_dword v[0:1], v8                                // 00000003CBA4: DC700000 00000800
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[88:89]                 // 00000003CBAC: D2080000 01610100
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[88:89]                 // 00000003CBB4: D2080002 01610102
	s_or_b64 s[10:11], s[6:7], s[10:11]                        // 00000003CBBC: 878A0A06
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[88:89]                 // 00000003CBC0: D2080004 01610104
	s_andn2_b64 exec, exec, s[10:11]                           // 00000003CBC8: 89FE0A7E
	s_cbranch_execz 15                                         // 00000003CBCC: BF88000F <EpCombineIntraNodeKernel_bf16_nop2p+0x3c0c>
	v_mov_b32_e32 v8, 0                                        // 00000003CBD0: 7E100280
	s_and_saveexec_b64 s[6:7], vcc                             // 00000003CBD4: BE86206A
	s_cbranch_execz 4                                          // 00000003CBD8: BF880004 <EpCombineIntraNodeKernel_bf16_nop2p+0x3bec>
	flat_load_dword v8, v[4:5]                                 // 00000003CBDC: DC500000 08000004
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003CBE4: BF8C0070
	v_add_f32_e32 v8, 0, v8                                    // 00000003CBE8: 02101080
	s_or_b64 exec, exec, s[6:7]                                // 00000003CBEC: 87FE067E
	s_and_saveexec_b64 s[6:7], s[4:5]                          // 00000003CBF0: BE862004
	s_cbranch_execz 65510                                      // 00000003CBF4: BF88FFE6 <EpCombineIntraNodeKernel_bf16_nop2p+0x3b90>
	flat_load_dword v9, v[2:3]                                 // 00000003CBF8: DC500000 09000002
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003CC00: BF8C0070
	v_add_f32_e32 v8, v8, v9                                   // 00000003CC04: 02101308
	s_branch 65505                                             // 00000003CC08: BF82FFE1 <EpCombineIntraNodeKernel_bf16_nop2p+0x3b90>
	s_or_b64 exec, exec, s[8:9]                                // 00000003CC0C: 87FE087E
	s_mov_b64 s[4:5], 0                                        // 00000003CC10: BE840180
	s_mov_b64 s[6:7], 0                                        // 00000003CC14: BE860180
	s_mov_b64 s[26:27], 0                                      // 00000003CC18: BE9A0180
	s_and_b64 vcc, exec, s[6:7]                                // 00000003CC1C: 86EA067E
	s_cbranch_vccz 3                                           // 00000003CC20: BF860003 <EpCombineIntraNodeKernel_bf16_nop2p+0x3c30>
	s_cmp_lg_u32 s46, 1                                        // 00000003CC24: BF07812E
	s_mov_b64 s[26:27], -1                                     // 00000003CC28: BE9A01C1
	s_cselect_b64 s[4:5], -1, 0                                // 00000003CC2C: 858480C1
	s_and_b64 vcc, exec, s[4:5]                                // 00000003CC30: 86EA047E
	s_cbranch_vccz 220                                         // 00000003CC34: BF8600DC <EpCombineIntraNodeKernel_bf16_nop2p+0x3fa8>
	v_readlane_b32 s4, v63, 7                                  // 00000003CC38: D2890004 00010F3F
	v_readlane_b32 s5, v63, 8                                  // 00000003CC40: D2890005 0001113F
	s_andn2_b64 vcc, exec, s[4:5]                              // 00000003CC48: 89EA047E
	s_mov_b64 s[4:5], 0                                        // 00000003CC4C: BE840180
	s_cbranch_vccnz 171                                        // 00000003CC50: BF8700AB <EpCombineIntraNodeKernel_bf16_nop2p+0x3f00>
	v_lshlrev_b32_e32 v20, 2, v42                              // 00000003CC54: 24285482
	v_lshl_add_u64 v[0:1], v[36:37], 0, v[20:21]               // 00000003CC58: D2080000 04510124
	s_mov_b64 s[6:7], 0                                        // 00000003CC60: BE860180
	s_branch 10                                                // 00000003CC64: BF82000A <EpCombineIntraNodeKernel_bf16_nop2p+0x3c90>
	v_lshl_add_u64 v[2:3], s[4:5], 2, v[0:1]                   // 00000003CC68: D2080002 04010404
	s_add_u32 s4, s4, 64                                       // 00000003CC70: 8004C004
	s_addc_u32 s5, s5, 0                                       // 00000003CC74: 82058005
	s_add_u32 s6, s6, 1                                        // 00000003CC78: 80068106
	s_addc_u32 s7, s7, 0                                       // 00000003CC7C: 82078007
	s_cmp_eq_u64 s[6:7], s[70:71]                              // 00000003CC80: BF124606
	flat_store_dword v[2:3], v4 nt                             // 00000003CC84: DC720000 00000402
	s_cbranch_scc1 156                                         // 00000003CC8C: BF85009C <EpCombineIntraNodeKernel_bf16_nop2p+0x3f00>
	v_mov_b32_e32 v4, 0                                        // 00000003CC90: 7E080280
	v_mov_b32_e32 v5, v29                                      // 00000003CC94: 7E0A031D
	s_mov_b64 s[8:9], s[74:75]                                 // 00000003CC98: BE88014A
	s_branch 6                                                 // 00000003CC9C: BF820006 <EpCombineIntraNodeKernel_bf16_nop2p+0x3cb8>
	s_or_b64 exec, exec, s[10:11]                              // 00000003CCA0: 87FE0A7E
	s_add_u32 s8, s8, -8                                       // 00000003CCA4: 8008C808
	s_addc_u32 s9, s9, -1                                      // 00000003CCA8: 8209C109
	s_cmp_lg_u64 s[8:9], 0                                     // 00000003CCAC: BF138008
	v_add_u32_e32 v5, 64, v5                                   // 00000003CCB0: 680A0AC0
	s_cbranch_scc0 120                                         // 00000003CCB4: BF840078 <EpCombineIntraNodeKernel_bf16_nop2p+0x3e98>
	ds_read_b64 v[2:3], v5                                     // 00000003CCB8: D8EC0000 02000005
	s_waitcnt lgkmcnt(0)                                       // 00000003CCC0: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[2:3]                            // 00000003CCC4: 7DDA0480
	s_and_saveexec_b64 s[10:11], vcc                           // 00000003CCC8: BE8A206A
	s_cbranch_execz 8                                          // 00000003CCCC: BF880008 <EpCombineIntraNodeKernel_bf16_nop2p+0x3cf0>
	v_lshl_add_u64 v[2:3], s[4:5], 2, v[2:3]                   // 00000003CCD0: D2080002 04090404
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[20:21]                 // 00000003CCD8: D2080002 04510102
	flat_load_dword v2, v[2:3] nt                              // 00000003CCE0: DC520000 02000002
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003CCE8: BF8C0070
	v_add_f32_e32 v4, v4, v2                                   // 00000003CCEC: 02080504
	s_or_b64 exec, exec, s[10:11]                              // 00000003CCF0: 87FE0A7E
	ds_read_b64 v[2:3], v5 offset:8                            // 00000003CCF4: D8EC0008 02000005
	s_waitcnt lgkmcnt(0)                                       // 00000003CCFC: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[2:3]                            // 00000003CD00: 7DDA0480
	s_and_saveexec_b64 s[10:11], vcc                           // 00000003CD04: BE8A206A
	s_cbranch_execz 8                                          // 00000003CD08: BF880008 <EpCombineIntraNodeKernel_bf16_nop2p+0x3d2c>
	v_lshl_add_u64 v[2:3], s[4:5], 2, v[2:3]                   // 00000003CD0C: D2080002 04090404
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[20:21]                 // 00000003CD14: D2080002 04510102
	flat_load_dword v2, v[2:3] nt                              // 00000003CD1C: DC520000 02000002
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003CD24: BF8C0070
	v_add_f32_e32 v4, v4, v2                                   // 00000003CD28: 02080504
	s_or_b64 exec, exec, s[10:11]                              // 00000003CD2C: 87FE0A7E
	ds_read_b64 v[2:3], v5 offset:16                           // 00000003CD30: D8EC0010 02000005
	s_waitcnt lgkmcnt(0)                                       // 00000003CD38: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[2:3]                            // 00000003CD3C: 7DDA0480
	s_and_saveexec_b64 s[10:11], vcc                           // 00000003CD40: BE8A206A
	s_cbranch_execz 8                                          // 00000003CD44: BF880008 <EpCombineIntraNodeKernel_bf16_nop2p+0x3d68>
	v_lshl_add_u64 v[2:3], s[4:5], 2, v[2:3]                   // 00000003CD48: D2080002 04090404
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[20:21]                 // 00000003CD50: D2080002 04510102
	flat_load_dword v2, v[2:3] nt                              // 00000003CD58: DC520000 02000002
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003CD60: BF8C0070
	v_add_f32_e32 v4, v4, v2                                   // 00000003CD64: 02080504
	s_or_b64 exec, exec, s[10:11]                              // 00000003CD68: 87FE0A7E
	ds_read_b64 v[2:3], v5 offset:24                           // 00000003CD6C: D8EC0018 02000005
	s_waitcnt lgkmcnt(0)                                       // 00000003CD74: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[2:3]                            // 00000003CD78: 7DDA0480
	s_and_saveexec_b64 s[10:11], vcc                           // 00000003CD7C: BE8A206A
	s_cbranch_execz 8                                          // 00000003CD80: BF880008 <EpCombineIntraNodeKernel_bf16_nop2p+0x3da4>
	v_lshl_add_u64 v[2:3], s[4:5], 2, v[2:3]                   // 00000003CD84: D2080002 04090404
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[20:21]                 // 00000003CD8C: D2080002 04510102
	flat_load_dword v2, v[2:3] nt                              // 00000003CD94: DC520000 02000002
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003CD9C: BF8C0070
	v_add_f32_e32 v4, v4, v2                                   // 00000003CDA0: 02080504
	s_or_b64 exec, exec, s[10:11]                              // 00000003CDA4: 87FE0A7E
	ds_read_b64 v[2:3], v5 offset:32                           // 00000003CDA8: D8EC0020 02000005
	s_waitcnt lgkmcnt(0)                                       // 00000003CDB0: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[2:3]                            // 00000003CDB4: 7DDA0480
	s_and_saveexec_b64 s[10:11], vcc                           // 00000003CDB8: BE8A206A
	s_cbranch_execz 8                                          // 00000003CDBC: BF880008 <EpCombineIntraNodeKernel_bf16_nop2p+0x3de0>
	v_lshl_add_u64 v[2:3], s[4:5], 2, v[2:3]                   // 00000003CDC0: D2080002 04090404
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[20:21]                 // 00000003CDC8: D2080002 04510102
	flat_load_dword v2, v[2:3] nt                              // 00000003CDD0: DC520000 02000002
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003CDD8: BF8C0070
	v_add_f32_e32 v4, v4, v2                                   // 00000003CDDC: 02080504
	s_or_b64 exec, exec, s[10:11]                              // 00000003CDE0: 87FE0A7E
	ds_read_b64 v[2:3], v5 offset:40                           // 00000003CDE4: D8EC0028 02000005
	s_waitcnt lgkmcnt(0)                                       // 00000003CDEC: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[2:3]                            // 00000003CDF0: 7DDA0480
	s_and_saveexec_b64 s[10:11], vcc                           // 00000003CDF4: BE8A206A
	s_cbranch_execz 8                                          // 00000003CDF8: BF880008 <EpCombineIntraNodeKernel_bf16_nop2p+0x3e1c>
	v_lshl_add_u64 v[2:3], s[4:5], 2, v[2:3]                   // 00000003CDFC: D2080002 04090404
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[20:21]                 // 00000003CE04: D2080002 04510102
	flat_load_dword v2, v[2:3] nt                              // 00000003CE0C: DC520000 02000002
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003CE14: BF8C0070
	v_add_f32_e32 v4, v4, v2                                   // 00000003CE18: 02080504
	s_or_b64 exec, exec, s[10:11]                              // 00000003CE1C: 87FE0A7E
	ds_read_b64 v[2:3], v5 offset:48                           // 00000003CE20: D8EC0030 02000005
	s_waitcnt lgkmcnt(0)                                       // 00000003CE28: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[2:3]                            // 00000003CE2C: 7DDA0480
	s_and_saveexec_b64 s[10:11], vcc                           // 00000003CE30: BE8A206A
	s_cbranch_execz 8                                          // 00000003CE34: BF880008 <EpCombineIntraNodeKernel_bf16_nop2p+0x3e58>
	v_lshl_add_u64 v[2:3], s[4:5], 2, v[2:3]                   // 00000003CE38: D2080002 04090404
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[20:21]                 // 00000003CE40: D2080002 04510102
	flat_load_dword v2, v[2:3] nt                              // 00000003CE48: DC520000 02000002
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003CE50: BF8C0070
	v_add_f32_e32 v4, v4, v2                                   // 00000003CE54: 02080504
	s_or_b64 exec, exec, s[10:11]                              // 00000003CE58: 87FE0A7E
	ds_read_b64 v[2:3], v5 offset:56                           // 00000003CE5C: D8EC0038 02000005
	s_waitcnt lgkmcnt(0)                                       // 00000003CE64: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[2:3]                            // 00000003CE68: 7DDA0480
	s_and_saveexec_b64 s[10:11], vcc                           // 00000003CE6C: BE8A206A
	s_cbranch_execz 65419                                      // 00000003CE70: BF88FF8B <EpCombineIntraNodeKernel_bf16_nop2p+0x3ca0>
	v_lshl_add_u64 v[2:3], s[4:5], 2, v[2:3]                   // 00000003CE74: D2080002 04090404
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[20:21]                 // 00000003CE7C: D2080002 04510102
	flat_load_dword v2, v[2:3] nt                              // 00000003CE84: DC520000 02000002
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003CE8C: BF8C0070
	v_add_f32_e32 v4, v4, v2                                   // 00000003CE90: 02080504
	s_branch 65410                                             // 00000003CE94: BF82FF82 <EpCombineIntraNodeKernel_bf16_nop2p+0x3ca0>
	s_andn2_b64 vcc, exec, s[76:77]                            // 00000003CE98: 89EA4C7E
	s_mov_b64 s[8:9], s[78:79]                                 // 00000003CE9C: BE88014E
	v_mov_b32_e32 v5, v52                                      // 00000003CEA0: 7E0A0334
	s_cbranch_vccz 7                                           // 00000003CEA4: BF860007 <EpCombineIntraNodeKernel_bf16_nop2p+0x3ec4>
	s_branch 65391                                             // 00000003CEA8: BF82FF6F <EpCombineIntraNodeKernel_bf16_nop2p+0x3c68>
	s_or_b64 exec, exec, s[10:11]                              // 00000003CEAC: 87FE0A7E
	s_add_u32 s8, s8, -1                                       // 00000003CEB0: 8008C108
	s_addc_u32 s9, s9, -1                                      // 00000003CEB4: 8209C109
	s_cmp_lg_u64 s[8:9], 0                                     // 00000003CEB8: BF138008
	v_add_u32_e32 v5, 8, v5                                    // 00000003CEBC: 680A0A88
	s_cbranch_scc0 65385                                       // 00000003CEC0: BF84FF69 <EpCombineIntraNodeKernel_bf16_nop2p+0x3c68>
	ds_read_b64 v[2:3], v5                                     // 00000003CEC4: D8EC0000 02000005
	s_waitcnt lgkmcnt(0)                                       // 00000003CECC: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[2:3]                            // 00000003CED0: 7DDA0480
	s_and_saveexec_b64 s[10:11], vcc                           // 00000003CED4: BE8A206A
	s_cbranch_execz 65524                                      // 00000003CED8: BF88FFF4 <EpCombineIntraNodeKernel_bf16_nop2p+0x3eac>
	v_lshl_add_u64 v[2:3], s[4:5], 2, v[2:3]                   // 00000003CEDC: D2080002 04090404
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[20:21]                 // 00000003CEE4: D2080002 04510102
	flat_load_dword v2, v[2:3] nt                              // 00000003CEEC: DC520000 02000002
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003CEF4: BF8C0070
	v_add_f32_e32 v4, v4, v2                                   // 00000003CEF8: 02080504
	s_branch 65515                                             // 00000003CEFC: BF82FFEB <EpCombineIntraNodeKernel_bf16_nop2p+0x3eac>
	v_mov_b32_e32 v1, s5                                       // 00000003CF00: 7E020205
	v_or_b32_e32 v0, s4, v42                                   // 00000003CF04: 28005404
	v_cmp_gt_u64_e32 vcc, s[46:47], v[0:1]                     // 00000003CF08: 7DD8002E
	s_and_saveexec_b64 s[4:5], vcc                             // 00000003CF0C: BE84206A
	s_cbranch_execz 35                                         // 00000003CF10: BF880023 <EpCombineIntraNodeKernel_bf16_nop2p+0x3fa0>
	s_mov_b64 s[6:7], 0                                        // 00000003CF14: BE860180
	s_branch 10                                                // 00000003CF18: BF82000A <EpCombineIntraNodeKernel_bf16_nop2p+0x3f44>
	v_lshl_add_u64 v[2:3], v[0:1], 2, v[36:37]                 // 00000003CF1C: D2080002 04910500
	v_lshl_add_u64 v[0:1], v[0:1], 0, 64                       // 00000003CF24: D2080000 03010100
	v_cmp_le_u64_e32 vcc, s[46:47], v[0:1]                     // 00000003CF2C: 7DD6002E
	s_or_b64 s[6:7], vcc, s[6:7]                               // 00000003CF30: 8786066A
	flat_store_dword v[2:3], v4                                // 00000003CF34: DC700000 00000402
	s_andn2_b64 exec, exec, s[6:7]                             // 00000003CF3C: 89FE067E
	s_cbranch_execz 23                                         // 00000003CF40: BF880017 <EpCombineIntraNodeKernel_bf16_nop2p+0x3fa0>
	v_mov_b32_e32 v4, 0                                        // 00000003CF44: 7E080280
	v_mov_b32_e32 v5, v29                                      // 00000003CF48: 7E0A031D
	s_mov_b64 s[8:9], s[46:47]                                 // 00000003CF4C: BE88012E
	s_branch 6                                                 // 00000003CF50: BF820006 <EpCombineIntraNodeKernel_bf16_nop2p+0x3f6c>
	s_or_b64 exec, exec, s[10:11]                              // 00000003CF54: 87FE0A7E
	s_add_u32 s8, s8, -1                                       // 00000003CF58: 8008C108
	s_addc_u32 s9, s9, -1                                      // 00000003CF5C: 8209C109
	s_cmp_eq_u64 s[8:9], 0                                     // 00000003CF60: BF128008
	v_add_u32_e32 v5, 8, v5                                    // 00000003CF64: 680A0A88
	s_cbranch_scc1 65516                                       // 00000003CF68: BF85FFEC <EpCombineIntraNodeKernel_bf16_nop2p+0x3f1c>
	ds_read_b64 v[2:3], v5                                     // 00000003CF6C: D8EC0000 02000005
	s_waitcnt lgkmcnt(0)                                       // 00000003CF74: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[2:3]                            // 00000003CF78: 7DDA0480
	s_and_saveexec_b64 s[10:11], vcc                           // 00000003CF7C: BE8A206A
	s_cbranch_execz 65524                                      // 00000003CF80: BF88FFF4 <EpCombineIntraNodeKernel_bf16_nop2p+0x3f54>
	v_lshl_add_u64 v[2:3], v[0:1], 2, v[2:3]                   // 00000003CF84: D2080002 04090500
	flat_load_dword v2, v[2:3]                                 // 00000003CF8C: DC500000 02000002
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003CF94: BF8C0070
	v_add_f32_e32 v4, v4, v2                                   // 00000003CF98: 02080504
	s_branch 65517                                             // 00000003CF9C: BF82FFED <EpCombineIntraNodeKernel_bf16_nop2p+0x3f54>
	s_or_b64 exec, exec, s[4:5]                                // 00000003CFA0: 87FE047E
	s_mov_b64 s[26:27], 0                                      // 00000003CFA4: BE9A0180
	s_and_b64 vcc, exec, s[26:27]                              // 00000003CFA8: 86EA1A7E
	s_cbranch_vccz 62262                                       // 00000003CFAC: BF86F336 <EpCombineIntraNodeKernel_bf16_nop2p+0xc88>
	v_readlane_b32 s4, v63, 9                                  // 00000003CFB0: D2890004 0001133F
	v_readlane_b32 s5, v63, 10                                 // 00000003CFB8: D2890005 0001153F
	s_and_b64 exec, exec, s[4:5]                               // 00000003CFC0: 86FE047E
	s_cbranch_execz 62256                                      // 00000003CFC4: BF88F330 <EpCombineIntraNodeKernel_bf16_nop2p+0xc88>
	ds_read_b64 v[2:3], v29                                    // 00000003CFC8: D8EC0000 0200001D
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[30:31]               // 00000003CFD0: D2080000 04790122
	s_mov_b64 s[6:7], 0                                        // 00000003CFD8: BE860180
	v_lshl_add_u64 v[0:1], v[32:33], 2, v[0:1]                 // 00000003CFDC: D2080000 04010520
	v_mov_b64_e32 v[4:5], v[42:43]                             // 00000003CFE4: 7E08712A
	s_waitcnt lgkmcnt(0)                                       // 00000003CFE8: BF8CC07F
	v_cmp_ne_u64_e32 vcc, 0, v[2:3]                            // 00000003CFEC: 7DDA0480
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[30:31]                 // 00000003CFF0: D2080002 04790102
	s_branch 14                                                // 00000003CFF8: BF82000E <EpCombineIntraNodeKernel_bf16_nop2p+0x4034>
	s_or_b64 exec, exec, s[4:5]                                // 00000003CFFC: 87FE047E
	v_lshl_add_u64 v[4:5], v[4:5], 0, 64                       // 00000003D000: D2080004 03010104
	v_cmp_le_u64_e64 s[4:5], s[46:47], v[4:5]                  // 00000003D008: D0EB0004 0002082E
	flat_store_dword v[0:1], v6                                // 00000003D010: DC700000 00000600
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[88:89]                 // 00000003D018: D2080000 01610100
	s_or_b64 s[6:7], s[4:5], s[6:7]                            // 00000003D020: 87860604
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[88:89]                 // 00000003D024: D2080002 01610102
	s_andn2_b64 exec, exec, s[6:7]                             // 00000003D02C: 89FE067E
	s_cbranch_execz 62229                                      // 00000003D030: BF88F315 <EpCombineIntraNodeKernel_bf16_nop2p+0xc88>
	v_mov_b32_e32 v6, 0                                        // 00000003D034: 7E0C0280
	s_and_saveexec_b64 s[4:5], vcc                             // 00000003D038: BE84206A
	s_cbranch_execz 65519                                      // 00000003D03C: BF88FFEF <EpCombineIntraNodeKernel_bf16_nop2p+0x3ffc>
	flat_load_dword v6, v[2:3]                                 // 00000003D040: DC500000 06000002
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000003D048: BF8C0070
	v_add_f32_e32 v6, 0, v6                                    // 00000003D04C: 020C0C80
	s_branch 65514                                             // 00000003D050: BF82FFEA <EpCombineIntraNodeKernel_bf16_nop2p+0x3ffc>
	s_endpgm                                                   // 00000003D054: BF810000
	s_andn2_b64 vcc, exec, s[4:5]                              // 00000003D058: 89EA047E
	s_cbranch_vccz 62041                                       // 00000003D05C: BF86F259 <EpCombineIntraNodeKernel_bf16_nop2p+0x9c4>
	s_branch 62064                                             // 00000003D060: BF82F270 <EpCombineIntraNodeKernel_bf16_nop2p+0xa24>
	s_nop 0                                                    // 00000003D064: BF800000
	s_nop 0                                                    // 00000003D068: BF800000
	s_nop 0                                                    // 00000003D06C: BF800000
	s_nop 0                                                    // 00000003D070: BF800000
	s_nop 0                                                    // 00000003D074: BF800000
	s_nop 0                                                    // 00000003D078: BF800000
	s_nop 0                                                    // 00000003D07C: BF800000
	s_nop 0                                                    // 00000003D080: BF800000
	s_nop 0                                                    // 00000003D084: BF800000
	s_nop 0                                                    // 00000003D088: BF800000
	s_nop 0                                                    // 00000003D08C: BF800000
	s_nop 0                                                    // 00000003D090: BF800000
	s_nop 0                                                    // 00000003D094: BF800000
	s_nop 0                                                    // 00000003D098: BF800000
	s_nop 0                                                    // 00000003D09C: BF800000
	s_nop 0                                                    // 00000003D0A0: BF800000
	s_nop 0                                                    // 00000003D0A4: BF800000
	s_nop 0                                                    // 00000003D0A8: BF800000
	s_nop 0                                                    // 00000003D0AC: BF800000
	s_nop 0                                                    // 00000003D0B0: BF800000
	s_nop 0                                                    // 00000003D0B4: BF800000
	s_nop 0                                                    // 00000003D0B8: BF800000
	s_nop 0                                                    // 00000003D0BC: BF800000
	s_nop 0                                                    // 00000003D0C0: BF800000
	s_nop 0                                                    // 00000003D0C4: BF800000
	s_nop 0                                                    // 00000003D0C8: BF800000
	s_nop 0                                                    // 00000003D0CC: BF800000
	s_nop 0                                                    // 00000003D0D0: BF800000
	s_nop 0                                                    // 00000003D0D4: BF800000
	s_nop 0                                                    // 00000003D0D8: BF800000
	s_nop 0                                                    // 00000003D0DC: BF800000
	s_nop 0                                                    // 00000003D0E0: BF800000
	s_nop 0                                                    // 00000003D0E4: BF800000
	s_nop 0                                                    // 00000003D0E8: BF800000
	s_nop 0                                                    // 00000003D0EC: BF800000
	s_nop 0                                                    // 00000003D0F0: BF800000
	s_nop 0                                                    // 00000003D0F4: BF800000
	s_nop 0                                                    // 00000003D0F8: BF800000
	s_nop 0                                                    // 00000003D0FC: BF800000
