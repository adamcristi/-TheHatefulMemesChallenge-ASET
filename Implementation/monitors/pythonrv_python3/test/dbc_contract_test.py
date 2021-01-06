# -*- coding: utf-8 -*-
import unittest

from pythonrv_python3 import dbc

class TestOnFunctions(unittest.TestCase):
    def test_return_value(self):
        def pre(x):
            return 17
        def post(x):
            return 'adsf'
        @dbc.contract(pre=pre, post=post)
        def m(x):
            return x+2

        res = m(3)
        self.assertEqual(res, 5)

    def test_pre(self):
        def pa():
            pa.val += 'a'
        pa.val = 'q'

        @dbc.contract(pre=pa)
        def m():
            pass

        m(); m(); m()
        self.assertEqual(pa.val, 'qaaa')

    def test_post(self):
        def pa():
            pa.val += 'a'
        pa.val = 'q'

        @dbc.contract(post=pa)
        def m():
            pass

        m(); m(); m()
        self.assertEqual(pa.val, 'qaaa')

    def test_iterables(self):
        def a(self):
            self.val += 'a'
        def b(self):
            self.val += 'b'
        def c(self):
            self.val += 'c'

        class Aloha(object):
            @dbc.contract(pre=(a, b), post=c)
            def m(self):
                self.val += 'm'

            @dbc.contract(pre=(), post=())
            def s(self):
                self.val += 's'

            @dbc.contract(pre=(a,(b,a),b), post=(c,a,((b))))
            def n(self):
                self.val += 'n'

        obj = Aloha()
        obj.val = 'q'
        obj.m()
        self.assertEqual(obj.val, 'qabmc')

        obj.val = 'q'
        obj.s()
        self.assertEqual(obj.val, 'qs')

        obj.val = 'q'
        obj.n()
        self.assertEqual(obj.val, 'qababncab')

    def test_requires_ensure(self):
        def a(self):
            self.val += 'a'
        def b(self):
            self.val += 'b'
        def c(self):
            self.val += 'c'

        class Aloha(object):
            @dbc.contract(pre=(a), requires=b, post=c)
            def m(self):
                self.val += 'm'

            @dbc.contract(requires=(), ensures=())
            def s(self):
                self.val += 's'

            @dbc.contract(requires=(a,(b,a),b), ensures=(c,a,((b))))
            def n(self):
                self.val += 'n'

        obj = Aloha()
        obj.val = 'q'
        obj.m()
        self.assertEqual(obj.val, 'qabmc')

        obj.val = 'q'
        obj.s()
        self.assertEqual(obj.val, 'qs')

        obj.val = 'q'
        obj.n()
        self.assertEqual(obj.val, 'qababncab')

    def test_arguments(self):
        def pa(*args, **kwargs):
            raise ValueError("%s %s" % (args, kwargs))
        def pb(x):
            raise ValueError(x)

        @dbc.contract(pre=pa)
        def m(x, y):
            return x + y
        with self.assertRaises(ValueError) as e:
            m(2,3)
        self.assertEqual(e.exception.message, "(2, 3) {}")

        @dbc.contract(post=pa)
        def n(x, *args, **kwargs):
            return x + kwargs['y']
        with self.assertRaises(ValueError) as e:
            n(2, 17, 20, y=4)
        self.assertEqual(e.exception.message, "(2, 17, 20) {'y': 4}")

        @dbc.contract(pre=pb)
        def o(z):
            return z+1
        with self.assertRaises(ValueError) as e:
            o(2)
        self.assertEqual(e.exception.message, 2)

        @dbc.contract(post=pb)
        def p(z):
            return z+1
        with self.assertRaises(ValueError) as e:
            p(2)
        self.assertEqual(e.exception.message, 2)

    def test_argument_mutability(self):
        def p(x={}):
            x['val'] -= 17

        @dbc.contract(pre=p, post=p)
        def m(x):
            return x['val']

        d = {'val': 17}
        self.assertEqual(m(d), 0)
        self.assertEqual(d['val'], -17)

    def test_func_attrib(self):
        def p(*args, **kwargs):
            pass

        @dbc.contract(pre=p, post=p)
        def m(x):
            m.val = x
        m(7)
        self.assertEqual(m.val, 7)

        @dbc.contract(pre=p, post=p)
        def n(x):
            n.val += x
        n.val = 1
        n(2); n(3); n(10)
        self.assertEqual(n.val, 16)

    def test_explicit_call(self):
        def p(x):
            p.val = 'x' if not hasattr(p, 'val') else p.val + 'y'
        def m(x):
            return x

        n = dbc.contract(pre=p, post=p)(m)
        res = n(7)
        self.assertEqual(res, 7)
        self.assertEqual(p.val, 'xy')


class TestOnClassFunctions(unittest.TestCase):
    def test_method_attachment(self):
        def p(self):
            self.val += 'p'

        class Aloha():
            def q(self):
                self.val += 'q'

            @dbc.contract(pre=p)
            def m(self):
                return 'm'
            @dbc.contract(post=q)
            def n(self):
                return 'n'
            @dbc.contract(pre=q, post=p)
            def x(self):
                return 'x'

        obj = Aloha()
        obj.val = 'a'

        res = obj.m();
        self.assertEqual(res, 'm')
        self.assertEqual(obj.val, 'ap')

        res = obj.n();
        self.assertEqual(res, 'n')
        self.assertEqual(obj.val, 'apq')

        res = obj.x();
        self.assertEqual(res, 'x')
        self.assertEqual(obj.val, 'apqqp')

    def test_class_method_attachment(self):
        def p(x):
            x.val += 'p'

        class Aloha():
            def q(x):
                x.val += 'q'

            @classmethod
            def a(cls):
                cls.val = 'a'
                return 'a'

            @dbc.contract(pre=p)
            @classmethod
            def m(cls):
                return 'm'
            @dbc.contract(post=q)
            @classmethod
            def n(cls):
                return 'n'
            @dbc.contract(pre=q, post=p)
            @classmethod
            def x(klass):
                return 'x'

        res = Aloha.a();
        self.assertEqual(res, 'a')
        self.assertEqual(Aloha.val, 'a')

        res = Aloha.m();
        self.assertEqual(res, 'm')
        self.assertEqual(Aloha.val, 'ap')

        res = Aloha.n();
        self.assertEqual(res, 'n')
        self.assertEqual(Aloha.val, 'apq')

        res = Aloha.x();
        self.assertEqual(res, 'x')
        self.assertEqual(Aloha.val, 'apqqp')

    def test_static_method_attachment(self):
        def p():
            Aloha.val += 'p'

        class Aloha():
            def q():
                Aloha.val += 'q'

            @staticmethod
            def a():
                Aloha.val = 'a'
                return 'a'

            @dbc.contract(pre=p)
            @staticmethod
            def m():
                return 'm'
            @dbc.contract(post=q)
            @staticmethod
            def n():
                return 'n'
            @dbc.contract(pre=q, post=p)
            @staticmethod
            def x():
                return 'x'

        res = Aloha.a();
        self.assertEqual(res, 'a')
        self.assertEqual(Aloha.val, 'a')

        res = Aloha.m();
        self.assertEqual(res, 'm')
        self.assertEqual(Aloha.val, 'ap')

        res = Aloha.n();
        self.assertEqual(res, 'n')
        self.assertEqual(Aloha.val, 'apq')

        res = Aloha.x();
        self.assertEqual(res, 'x')
        self.assertEqual(Aloha.val, 'apqqp')
