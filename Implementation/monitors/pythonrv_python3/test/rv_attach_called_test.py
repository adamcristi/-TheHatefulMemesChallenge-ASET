# -*- coding: utf-8 -*-
import unittest

from pythonrv_python3 import rv

# these tests test that (1) @rv.monitor works as expected - that it can attach
# to the desired functions and (2) that .called and .active_monitor work

def t_attrib():
    return -1

def t_one():
    return 1
def t_two():
    return 2

class TestWhenAttach(unittest.TestCase):
    def test_attach_before(self):
        class M(object):
            def m(self):
                raise ValueError('buffy')

        @rv.monitor(m=M.m)
        @rv.spec_save_automatically()
        def spec(event):
            raise ValueError('willow')

        a = M()
        with self.assertRaises(ValueError) as e:
            a.m()
        self.assertEqual(e.exception.message, 'willow')

    def test_attach_after(self):
        class M(object):
            def m(self, x):
                if x == 0:
                    raise ValueError('buffy')

        @rv.monitor(m=M.m)
        @rv.spec_save_automatically()
        def spec(event):
            raise ValueError('willow')

        a = M()
        with self.assertRaises(ValueError) as e:
            a.m(0)
        self.assertEqual(e.exception.message, 'buffy')
        with self.assertRaises(ValueError) as e:
            a.m(1)
        self.assertEqual(e.exception.message, 'willow')

    def test_attach_default_should_be_before(self):
        class M(object):
            def m(self):
                raise ValueError('buffy')

        @rv.monitor(m=M.m)
        def spec(event):
            raise ValueError('willow')

        a = M()
        with self.assertRaises(ValueError) as e:
            a.m()
        self.assertEqual(e.exception.message, 'willow')


class TestCalledAndActive(unittest.TestCase):
    def test_attributes(self):
        t_attrib.var = 'x'
        self.assertEqual(t_attrib.var, 'x')

        @rv.monitor(t=t_attrib)
        def spec(event):
            pass

        self.assertEqual(t_attrib.var, 'x')
        t_attrib.var = 'y'
        self.assertEqual(t_attrib.var, 'y')

    def test_one_function(self):
        self.assertEqual(t_one(), 1)

        @rv.monitor(test=t_one)
        def spec(event):
            self.assertTrue(event.fn.test.called)
            self.assertEqual(event.called_function, event.fn.test)
            raise ValueError("test called")

        with self.assertRaises(ValueError) as e:
            t_one()
        self.assertEqual(e.exception.message, "test called")

    def test_one_function_two_specs(self):
        self.assertEqual(t_two(), 2)

        enable_first = True

        @rv.monitor(test=t_two)
        def spec(event):
            self.assertTrue(event.fn.test.called)
            self.assertEqual(event.called_function, event.fn.test)
            if enable_first:
                raise ValueError("test called")

        with self.assertRaises(ValueError) as e:
            t_two()
        self.assertEqual(e.exception.message, "test called")

        enable_first = False
        self.assertEqual(t_two(), 2)

        @rv.monitor(test=t_two)
        def spec2(event):
            self.assertTrue(event.fn.test.called)
            self.assertEqual(event.called_function, event.fn.test)
            raise ValueError("test called2")

        with self.assertRaises(ValueError) as e:
            t_two()
        self.assertEqual(e.exception.message, "test called2")

        enable_first = True
        with self.assertRaises(ValueError) as e:
            t_two()
        self.assertEqual(e.exception.message, "test called")

    def test_one_method(self):
        class M(object):
            def a(self):
                return 1

        @rv.monitor(a=M.a)
        def spec(event):
            self.assertTrue(event.fn.a.called)
            self.assertEqual(event.called_function, event.fn.a)
            raise ValueError("a called")

        m = M()
        with self.assertRaises(ValueError) as e:
            m.a()
        self.assertEqual(e.exception.message, "a called")

    def test_two_methods(self):
        class M(object):
            def a(self):
                return 1
            def b(self):
                return 0
            def c(self):
                return -1

        @rv.monitor(a=M.a, b=M.b)
        def spec(event):
            self.assertTrue(event.fn.a.called or event.fn.b.called)
            if event.fn.a.called:
                self.assertEqual(event.called_function, event.fn.a)
                raise ValueError("a called")
            if event.fn.b.called:
                self.assertEqual(event.called_function, event.fn.b)
                raise ValueError("b called")

        m = M()
        with self.assertRaises(ValueError) as e:
            m.a()
        self.assertEqual(e.exception.message, "a called")

        with self.assertRaises(ValueError) as e:
            m.b()
        self.assertEqual(e.exception.message, "b called")

        self.assertEqual(m.c(), -1)

    def test_two_methods_two_specs(self):
        class M(object):
            def a(self):
                return 1
            def b(self):
                return 0
            def c(self):
                return -1

        enable_first = True
        @rv.monitor(a=M.a, b=M.b)
        def spec(event):
            self.assertTrue(event.fn.a.called or event.fn.b.called)
            if event.fn.a.called:
                self.assertEqual(event.called_function, event.fn.a)
                if enable_first:
                    raise ValueError("a called")
            if event.fn.b.called:
                self.assertEqual(event.called_function, event.fn.b)
                if enable_first:
                    raise ValueError("b called")

        m = M()
        with self.assertRaises(ValueError) as e:
            m.a()
        self.assertEqual(e.exception.message, "a called")

        with self.assertRaises(ValueError) as e:
            m.b()
        self.assertEqual(e.exception.message, "b called")

        self.assertEqual(m.c(), -1)

        enable_first = False
        self.assertEqual(m.a(), 1)
        self.assertEqual(m.b(), 0)
        self.assertEqual(m.c(), -1)

        @rv.monitor(a=M.a, b=M.b)
        def spec2(event):
            self.assertTrue(event.fn.a.called or event.fn.b.called)
            if event.fn.a.called:
                self.assertEqual(event.called_function, event.fn.a)
                raise ValueError("a called2")
            if event.fn.b.called:
                self.assertEqual(event.called_function, event.fn.b)
                raise ValueError("b called2")

        with self.assertRaises(ValueError) as e:
            m.a()
        self.assertEqual(e.exception.message, "a called2")

        with self.assertRaises(ValueError) as e:
            m.b()
        self.assertEqual(e.exception.message, "b called2")

        self.assertEqual(m.c(), -1)

        enable_first = True
        with self.assertRaises(ValueError) as e:
            m.a()
        self.assertEqual(e.exception.message, "a called")

        with self.assertRaises(ValueError) as e:
            m.b()
        self.assertEqual(e.exception.message, "b called")

        self.assertEqual(m.c(), -1)

class TestClassmethodStaticmethod(unittest.TestCase):
    def test_classmethod(self):
        class M(object):
            @classmethod
            def a(cls):
                return 'a'

        self.assertEqual(M.a(), 'a')

        enable_first = True
        @rv.monitor(a=M.a)
        def spec(event):
            if enable_first:
                raise ValueError("in spec")

        with self.assertRaises(ValueError) as e:
            M.a()
        self.assertEqual(e.exception.message, "in spec")

        enable_first = False
        @rv.monitor(a=M.a)
        def spec2(event):
            raise ValueError("in spec2")

        with self.assertRaises(ValueError) as e:
            M.a()
        self.assertEqual(e.exception.message, "in spec2")

        enable_first = True
        with self.assertRaises(ValueError) as e:
            M.a()
        self.assertEqual(e.exception.message, "in spec")

    def test_staticmethod(self):
        # static methods unfortunately require special treatment
        # TODO: fix
        class M(object):
            @staticmethod
            def a():
                return 'a'

        self.assertEqual(M.a(), 'a')

        enable_first = True
        @rv.monitor(a=(M, M.a))
        def spec(event):
            if enable_first:
                raise ValueError("in spec")

        with self.assertRaises(ValueError) as e:
            M.a()
        self.assertEqual(e.exception.message, "in spec")

        enable_first = False
        @rv.monitor(a=(M, M.a))
        def spec2(event):
            raise ValueError("in spec2")

        with self.assertRaises(ValueError) as e:
            M.a()
        self.assertEqual(e.exception.message, "in spec2")

        enable_first = True
        with self.assertRaises(ValueError) as e:
            M.a()
        self.assertEqual(e.exception.message, "in spec")

class TestOnObject(unittest.TestCase):
    def test_on_object(self):
        class M(object):
            def m(self):
                return 'm'

        a = M()
        b = M()

        self.assertEqual(a.m(), 'm')
        self.assertEqual(b.m(), 'm')

        # attach spec on a
        @rv.monitor(m=a.m)
        def spec(event):
            raise ValueError("m called on a")

        with self.assertRaises(ValueError) as e:
            a.m()
        self.assertEqual(e.exception.message, "m called on a")
        self.assertEqual(b.m(), 'm')
        self.assertEqual(M().m(), 'm')

        # attach spec on c with tuple format
        c = M()
        @rv.monitor(m=(c, c.m))
        def spec2(event):
            raise ValueError("m called on c")

        with self.assertRaises(ValueError) as e:
            c.m()
        self.assertEqual(e.exception.message, "m called on c")
        self.assertEqual(b.m(), 'm')
        self.assertEqual(M().m(), 'm')

        # attach spec on d wiht tuple-string format
        d = M()
        @rv.monitor(m=(d, "m"))
        def spec3(event):
            raise ValueError("m called on d")

        with self.assertRaises(ValueError) as e:
            d.m()
        self.assertEqual(e.exception.message, "m called on d")
        self.assertEqual(b.m(), 'm')
        self.assertEqual(M().m(), 'm')
