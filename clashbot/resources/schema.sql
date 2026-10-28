PRAGMA journal_mode = WAL;
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE games (game_id TEXT PRIMARY KEY, split TEXT, own_deck TEXT, deck_status TEXT, own_deck_key TEXT,
  own_archetype TEXT, opponent_deck TEXT, opponent_archetype_guess TEXT, result TEXT, flags TEXT);
CREATE TABLE provenance (game_id TEXT PRIMARY KEY, video_id TEXT, match_index TEXT, creator TEXT,
  video_title TEXT, video_url TEXT, video_time TEXT);
CREATE TABLE states (vec_row INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, game_id TEXT, split TEXT, kind TEXT,
  t REAL, phase TEXT, match_seconds REAL, own_elixir REAL, deck_status TEXT, own_deck_key TEXT,
  action_type TEXT, action_card TEXT, action_lane TEXT, action_zone TEXT, action_key TEXT,
  verdict TEXT, elixir_trade REAL, train_eligible INTEGER, flags TEXT, state_text TEXT, context_text TEXT, record TEXT);
CREATE INDEX states_game ON states(game_id);
CREATE INDEX states_split ON states(split);
CREATE INDEX states_deck ON states(own_deck_key);
CREATE INDEX states_phase ON states(phase);
CREATE INDEX states_action_card ON states(action_card);
CREATE VIRTUAL TABLE states_fts USING fts5(id UNINDEXED, context_text);
CREATE TABLE cards (slug TEXT PRIMARY KEY, name TEXT, card_type TEXT, rarity TEXT, elixir_cost REAL, targets TEXT,
  attributes TEXT, stats_by_level TEXT);
CREATE TABLE kg_nodes (id TEXT PRIMARY KEY, type TEXT, name TEXT, props TEXT);
CREATE TABLE kg_edges (edge_id INTEGER PRIMARY KEY, src TEXT, rel TEXT, dst TEXT, props TEXT);
CREATE INDEX kg_edges_src ON kg_edges(src, rel);
CREATE INDEX kg_edges_dst ON kg_edges(dst, rel);
CREATE TABLE documents (doc_id TEXT PRIMARY KEY, type TEXT, title TEXT, text TEXT, cards TEXT, deck_key TEXT,
  archetype TEXT, game_id TEXT, split TEXT);
CREATE VIRTUAL TABLE documents_fts USING fts5(doc_id UNINDEXED, title, text);
