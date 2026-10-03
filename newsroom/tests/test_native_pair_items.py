"""Lossless ordered CURRENT rights inventory sharing, not audit or body caching."""
from copy import deepcopy
import json
import pytest
from newsroom.authority.canonical import canonical_json_bytes,digest_bytes
from newsroom.control_plane import native_progress_state as state
from newsroom.control_plane.store import connect


def _pair(size):
    return {'retrieval_binding':{'request':'bound'},'retrieval_rights_inventory':[
        {'revision_id':f'revision-{i}','ingest_id':f'ingest-{i}','state':'INCLUDED','reason':None,
         'document_digest':'sha256:'+f'{i:064x}','current_rights_digest':'sha256:'+'a'*64}for i in range(size)]}


def test_growing_inventory_shares_items_and_reconstructs_exact_pair_bytes(tmp_path):
    connection=connect(str(tmp_path/'pairs.sqlite3'))
    try:
        originals=[]
        for size in (2000,2001,2002):
            pair=_pair(size);raw=canonical_json_bytes(pair);digest=digest_bytes(raw)
            state.retain_pair(connection,pair,digest)
            originals.append((digest,raw))
        assert connection.execute('SELECT count(*) FROM native_current_inventory_items').fetchone()[0]==2002
        for digest,raw in originals:
            assert canonical_json_bytes(state.read_pair(connection,digest))==raw
        encoded_bytes=connection.execute('SELECT sum(length(CAST(pair_json AS BLOB))) FROM native_current_pairs').fetchone()[0]
        item_bytes=connection.execute('SELECT sum(length(CAST(item_json AS BLOB))) FROM native_current_inventory_items').fetchone()[0]
        assert encoded_bytes+item_bytes<sum(len(raw)for _digest,raw in originals)*0.6
        returned=state.read_pair(connection,originals[0][0]);returned['retrieval_rights_inventory'][0]['state']='MUTATED'
        assert canonical_json_bytes(state.read_pair(connection,originals[0][0]))==originals[0][1]
    finally:connection.close()


@pytest.mark.parametrize('mutation',('missing','item','order','reference','format'))
def test_pair_item_corruption_is_denied(tmp_path,mutation):
    connection=connect(str(tmp_path/'corrupt.sqlite3'))
    try:
        pair=_pair(3);digest=digest_bytes(canonical_json_bytes(pair));state.retain_pair(connection,pair,digest)
        raw=connection.execute('SELECT pair_json FROM native_current_pairs WHERE pair_digest=?',(digest,)).fetchone()[0]
        encoded=json.loads(raw)
        first=encoded['inventory_items'][0]
        if mutation=='missing':
            connection.execute('DELETE FROM native_current_pair_items WHERE item_id=?',(first,))
            connection.execute('DELETE FROM native_current_inventory_items WHERE item_id=?',(first,))
        elif mutation=='item':connection.execute('UPDATE native_current_inventory_items SET item_json=? WHERE item_id=?',('{}',first))
        else:
            if mutation=='order':encoded['inventory_items'].reverse()
            elif mutation=='reference':encoded['inventory_items'][0]='sha256:'+'0'*64
            else:encoded['schema']='unknown'
            connection.execute('UPDATE native_current_pairs SET pair_json=? WHERE pair_digest=?',(canonical_json_bytes(encoded).decode(),digest))
        with pytest.raises(ValueError):state.read_pair(connection,digest)
    finally:connection.close()


def _headed_legacy(connection):
    from newsroom.control_plane.native_progress import NativeRevisionJournal
    from newsroom.tests.test_native_graphiti import _native
    journal=NativeRevisionJournal(connection);unit=_native();journal.land((unit,))
    facts=_pair(3);journal.advance(unit.revision_id,stage='EVIDENCE_HOLD',facts=facts)
    digest=digest_bytes(canonical_json_bytes(facts))
    connection.execute('UPDATE native_current_pairs SET pair_json=? WHERE pair_digest=?',(canonical_json_bytes(facts).decode(),digest))
    connection.execute('DELETE FROM native_current_pair_inventories')
    connection.execute('DELETE FROM native_current_inventory_items')
    state.write_counts(connection);connection.commit()
    return unit,journal,digest,facts


def test_headed_pair_conversion_updates_same_root_and_reopens_exactly(tmp_path):
    from newsroom.control_plane.native_progress import NativeRevisionJournal
    connection=connect(str(tmp_path/'headed.sqlite3'))
    try:
        unit,journal,digest,facts=_headed_legacy(connection)
        head=connection.execute('SELECT * FROM native_current_heads').fetchone()
        logical=canonical_json_bytes(journal.current(unit.revision_id))
        assert state.compact_current_pairs(connection)=={'converted_pairs':1,'inventory_items':3}
        assert connection.execute('SELECT * FROM native_current_heads').fetchone()==head
        assert canonical_json_bytes(NativeRevisionJournal(connection).current(unit.revision_id))==logical
        assert state.compact_current_pairs(connection)['converted_pairs']==0
        state.require_ready(connection)
    finally:connection.close()


def test_conversion_failure_rolls_back_pair_and_count_upgrade(tmp_path,monkeypatch):
    connection=connect(str(tmp_path/'rollback.sqlite3'))
    try:
        _unit,_journal,_digest,_facts=_headed_legacy(connection)
        before={table:list(connection.execute(f'SELECT * FROM {table}'))for table in ('native_current_pairs','native_current_meta','native_current_inventory_items','native_current_pair_inventories','native_current_pair_items')}
        write=state.write_counts
        def interrupted(conn):
            write(conn);raise RuntimeError('interrupted before commit')
        monkeypatch.setattr(state,'write_counts',interrupted)
        with pytest.raises(RuntimeError,match='before commit'):state.compact_current_pairs(connection)
        assert not connection.in_transaction
        assert all(list(connection.execute(f'SELECT * FROM {table}'))==rows for table,rows in before.items())
    finally:connection.close()


def test_replacing_pair_collects_only_unreferenced_shared_items(tmp_path):
    from newsroom.control_plane.native_progress import NativeRevisionJournal
    from newsroom.tests.test_native_graphiti import _native
    connection=connect(str(tmp_path/'shared.sqlite3'))
    try:
        journal=NativeRevisionJournal(connection);first,second=_native(),_native('second')
        for unit,size in ((first,3),(second,2)):
            journal.land((unit,));journal.advance(unit.revision_id,stage='EVIDENCE_HOLD',facts=_pair(size))
        assert connection.execute('SELECT count(*) FROM native_current_inventory_items').fetchone()[0]==3
        journal.advance(first.revision_id,stage='EVIDENCE_HOLD',facts={'reason':'different'})
        assert connection.execute('SELECT count(*) FROM native_current_inventory_items').fetchone()[0]==2
        assert journal.current(second.revision_id)['facts']==_pair(2)
        journal.advance(second.revision_id,stage='EVIDENCE_HOLD',facts={'reason':'different'})
        assert connection.execute('SELECT count(*) FROM native_current_inventory_items').fetchone()[0]==0
        state.require_ready(connection)
    finally:connection.close()


def test_old_inventory_counts_cannot_hide_uncounted_items(tmp_path):
    connection=connect(str(tmp_path/'counts.sqlite3'))
    try:
        _unit,_journal,_digest,_facts=_headed_legacy(connection)
        version,raw,_digest=connection.execute('SELECT version,counts_json,digest FROM native_current_meta').fetchone()
        counts=json.loads(raw)
        for name in state._PAIR_ITEM_TABLES:counts.pop(name)
        connection.execute('UPDATE native_current_meta SET counts_json=?,digest=?',(canonical_json_bytes(counts).decode(),digest_bytes(canonical_json_bytes({'version':version,'counts':counts}))))
        state.require_ready(connection)
        connection.execute('INSERT INTO native_current_inventory_items(item_digest,item_json) VALUES(?,?)',('sha256:'+'0'*64,'{}'))
        with pytest.raises(ValueError,match='uncounted'):state.require_ready(connection)
    finally:connection.close()


def test_existing_inventory_beyond_document_contract_stays_exact_inline(tmp_path):
    connection=connect(str(tmp_path/'large.sqlite3'))
    try:
        pair=_pair(4097);raw=canonical_json_bytes(pair);digest=digest_bytes(raw)
        state.retain_pair(connection,pair,digest)
        assert connection.execute('SELECT pair_json FROM native_current_pairs WHERE pair_digest=?',(digest,)).fetchone()[0].encode()==raw
        assert connection.execute('SELECT count(*) FROM native_current_inventory_items').fetchone()[0]==0
        assert canonical_json_bytes(state.read_pair(connection,digest))==raw
    finally:connection.close()


@pytest.mark.parametrize('mutation',('missing-root','missing-member','extra-member','cross-pair','logical-digest'))
def test_selected_pair_owner_and_order_remain_bound(tmp_path,mutation):
    connection=connect(str(tmp_path/'bindings.sqlite3'))
    try:
        pair=_pair(2);digest=digest_bytes(canonical_json_bytes(pair));state.retain_pair(connection,pair,digest)
        other=_pair(3);other_digest=digest_bytes(canonical_json_bytes(other));state.retain_pair(connection,other,other_digest)
        root=connection.execute('SELECT inventory_id FROM native_current_pair_inventories WHERE pair_digest=?',(digest,)).fetchone()[0]
        ids=[row[0]for row in connection.execute('SELECT item_id FROM native_current_pair_items WHERE inventory_id=?',(root,))]
        extra=connection.execute('SELECT max(item_id) FROM native_current_inventory_items').fetchone()[0]
        if mutation=='missing-root':connection.execute('DELETE FROM native_current_pair_inventories WHERE inventory_id=?',(root,))
        elif mutation=='missing-member':connection.execute('DELETE FROM native_current_pair_items WHERE inventory_id=? AND item_id=?',(root,ids[0]))
        elif mutation=='extra-member':connection.execute('INSERT INTO native_current_pair_items VALUES(?,?)',(root,extra))
        else:
            encoded=json.loads(connection.execute('SELECT pair_json FROM native_current_pairs WHERE pair_digest=?',(digest,)).fetchone()[0])
            if mutation=='cross-pair':encoded['inventory_items'][0]=extra
            else:encoded['retrieval_binding']={'request':'changed'}
            connection.execute('UPDATE native_current_pairs SET pair_json=? WHERE pair_digest=?',(canonical_json_bytes(encoded).decode(),digest))
        with pytest.raises(ValueError):state.read_pair(connection,digest)
        assert state.read_pair(connection,other_digest)==other
    finally:connection.close()
